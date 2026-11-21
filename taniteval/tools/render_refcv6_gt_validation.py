#!/usr/bin/env python3
"""render_refcv6_gt_validation.py -- refcv6's agent GROUND TRUTH, validated: a reel + sanity stats.

PI (Sayed), 2026-09-26, after the step-38,000 boxes video, verbatim: *"the detected objects bounding
boxes are completelly messy, the model is detecting boxes where is nothing. We need to understand what
happening, and review training, architecture, wiring, training signal flow and validate gt. let begin
by the last one and have a second video image in our video with onyl the gt boxes amd their classes"*.

⭐ NO MODEL FORWARD. Nothing here builds, loads or runs a network. The GT is the trainer's own per-window
block (``V3Dataset._agent_item``, ``refc_v3_train.py:2764-2864``) on the eval cache; the trainer's
target set is ``refc_agents.visible_target_filter`` (``refc_agents.py:410-433``); the camera is the
model's own per-clip ``RigCamera`` built by the trainer's ``_build_rig_camera(cfg, args)`` from the
run's config (no weights), cross-checked against ``render_refcv3_video.CylProjector`` per clip.

Per frame (1920 x 1200):
  * camera A (top): the step-38,000 boxes video's camera image (model boxes + GT targets), cropped
    from its SAVED frame -- or, for a clip that video never rendered, the RAW camera image;
  * camera B (below): ONLY the GT, projected cuboids in class colours, each labelled class · range ·
    track number; GT the trainer's filter REMOVES is dashed and labelled "filtered: <reason>";
  * BEV A: the boxes video's detection panel (model + GT), cropped from its saved frame;
  * BEV B: ONLY the GT, a WIDE field (x -20..80 m, y +-32 m) with the trainer's target region
    (120 deg field ∩ decode box x 0..60 m, |y| <= 16 m) outlined; class colours + a 3-letter tag;
  * the frame's GT census and sanity flags.

Sanity stats (one JSON per clip + pooled, every number with its n): GT per window by class; range
histogram; fraction outside the 120 deg field; fraction removed by the filter per reason; per-class
L/W/H medians and outliers against LITERAL class norms; box-bottom height (cz - h/2) against the rig
ground plane; duplicate GT (rotated BEV IoU > 0.5 within one frame); provenance with file:line.

Controls (must read known values): the reason decomposition must reproduce the trainer's own
``visible_target_filter`` mask exactly (0 mismatches); RigCamera and CylProjector must agree on every
projected GT corner (<= 0.01 px); every GT box drawn in the BEV must invert to its own metric corners
(<= half a pixel); bev_iou must read 1 on a box with itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULTS = {
    "repo": "C:/Users/Admin/ev6_82c2331",
    "battery_code": "C:/Users/Admin/ev6_battery/code",
    "config": "D:/refcv6_eval_kit/ckpt_final/config.json",
    "expect_config_md5": "a3193a4685ce0d07a6ae6b89fdc994a8",
    "boxes_frames": "C:/Users/Admin/qland/work/mapvid/frames_refcv6_boxes_step38000",
    "boxes_raw": ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                  "2026-09-26-refcv6-boxes-video/raw"),
    "clips": "0191487845ef,9f8bedcfb9de,34765c024267",
    "work": "C:/Users/Admin/qland/work/gtval",
}
TOOLS = Path(__file__).resolve().parent

# ---- the trainer's target region (LITERALS, asserted against the trainer at run time) ---------- #
FOV_HALF_DEG = 60.0                 # refc_agents.FOV_HALF_ANGLE_RAD
DECODE_X_M, DECODE_Y_M = 60.0, 16.0  # agent_slots.SlotDecodeRanges (x_fwd_m, y_half_m)
REASONS = ("behind ego (x < 0)", "outside the 120 deg field", "beyond 60 m ahead", "|y| > 16 m")
REASON_SHORT = ("behind", "outside FOV", ">60 m", "|y|>16 m")
CLASS_TAG = {"automobile": "car", "heavy_truck": "trk", "bus": "bus", "other_vehicle": "veh",
             "trailer": "trl", "person": "ped", "rider": "rdr", "stroller": "str", "animal": "ani",
             "protruding_object": "obj"}
#: LITERAL class norms (metres): (L_min, L_max, W_min, W_max, H_min, H_max). Independently authored
#: engineering bounds, NOT fit to this data. The PI's two examples are pinned exactly: a person longer
#: than 2 m, a car (automobile) taller than 3 m. EU limits (Directive 96/53/EC): width 2.55 m
#: (+ mirrors), height 4.0 m, rigid bus 13.5 m / articulated 18.75 m; 4.6 m / 25 m leave margin.
NORMS = {
    "automobile": (2.5, 6.5, 1.3, 2.6, 1.0, 3.0),
    "heavy_truck": (4.0, 22.0, 1.8, 3.2, 1.8, 4.6),
    "bus": (6.0, 25.0, 2.0, 3.2, 2.4, 4.6),
    "other_vehicle": (0.8, 22.0, 0.4, 3.5, 0.4, 4.6),
    "trailer": (1.0, 20.0, 0.8, 3.2, 0.3, 4.6),
    "person": (0.1, 2.0, 0.1, 1.5, 0.8, 2.3),
    "rider": (0.3, 3.0, 0.2, 1.5, 0.8, 2.5),
    "stroller": (0.3, 1.8, 0.2, 1.2, 0.4, 1.6),
    "animal": (0.1, 3.0, 0.1, 1.8, 0.1, 2.5),
    "protruding_object": None,          # airborne / odd by definition: reported, never flagged
}
FLOAT_M = 0.5                       # |cz - h/2| above/below the rig ground plane that is flagged
DUP_IOU = 0.5
#: the Master Mind, 2026-09-27 ~01:00, after the box-head audit's verdict (its code/patch_draw_cuboid/):
#: the projection was verified independently against LiDAR; the "one surface" boxes were a DRAWING defect,
#: fixed by the audit's sampled, border-clipped edges; heavy_truck GT is sunk in the label itself
REVIEW_CAPTION = ("3-D overlay verified independently against LiDAR; heavy_truck GT boxes are sunk "
                  "(label defect)")
RANGE_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, float("inf"))

# ---- layout ------------------------------------------------------------------------------------ #
W_TOT, H_TOT, PAD = 1920, 1200, 16
H_BAN = 76
CAM_W, CAM_H = 1024, 416
Y_CAMA, Y_CAMB = 84, 84 + CAM_H + 10
X_R = PAD + CAM_W + 16                         # right column
BEVA_BOX = (1156, 512, 1522, 1182)             # the boxes video's detection panel, in ITS frame
CAMA_BOX = (16, 84, 1040, 500)                 # the boxes video's camera image, in ITS frame
BEVB_X0, BEVB_X1, BEVB_Y = -20.0, 80.0, 32.0   # metres: x from, x to, |y|
BEVB_PPM = 6.0                                 # px per metre
BEVB_W, BEVB_H = int(2 * BEVB_Y * BEVB_PPM), int((BEVB_X1 - BEVB_X0) * BEVB_PPM)   # 384 x 600
C_BG, C_PANEL, C_BAN = (9, 12, 17), (16, 21, 29), (15, 20, 28)
C_FG, C_DIM, C_WARN, C_GT, C_BAD = (233, 238, 245), (140, 152, 168), (245, 180, 90), (110, 231, 138), \
    (255, 90, 90)


# ============================================================================================ #
# PURE functions (unit-tested without the stack)                                                 #
# ============================================================================================ #
def filter_reasons(cx, cy, valid) -> dict:
    """The trainer's ``filter_targets_to_visible`` (refc_agents.py:336-406) with the decode box of
    ``visible_target_filter`` (:410-433), DECOMPOSED into named reasons. ``keep`` must equal the
    trainer's own mask -- the run asserts it on every window."""
    cx, cy, valid = np.asarray(cx, float), np.asarray(cy, float), np.asarray(valid, bool)
    behind = cx < 0.0
    out_fov = (np.arctan2(np.abs(cy), cx) > math.radians(FOV_HALF_DEG)) & ~behind
    beyond_x = cx > DECODE_X_M
    beyond_y = np.abs(cy) > DECODE_Y_M
    stack = np.stack([behind, out_fov, beyond_x, beyond_y]) & valid[None]
    keep = valid & ~stack.any(0)
    primary = np.full(cx.shape, -1, dtype=np.int64)
    for k in range(3, -1, -1):                       # the FIRST reason in REASONS order wins
        primary[stack[k]] = k
    return {"keep": keep, "reasons": stack, "primary": primary}


def norm_flags(cls_name: str, l: float, w: float, h: float | None) -> list:
    nm = NORMS.get(cls_name)
    if nm is None:
        return []
    out = []
    for tag, v, lo, hi in (("L", l, nm[0], nm[1]), ("W", w, nm[2], nm[3]),
                           ("H", h, nm[4], nm[5])):
        if v is None or not np.isfinite(v):
            continue
        if v < lo:
            out.append(f"{tag} {v:.2f} m < {lo:g}")
        elif v > hi:
            out.append(f"{tag} {v:.2f} m > {hi:g}")
    return out


def box_corners(cx, cy, l, w, yaw) -> np.ndarray:
    """[4, 2] footprint, ordered around the box (front-left, front-right, rear-right, rear-left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]], float)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([cx, cy])


def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: convex ``subject`` clipped by convex ``clipper`` (both CCW or both CW)."""
    def inside(p, a, b, sgn):
        return sgn * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-12

    def inter(p, q, a, b):
        dc = a - b
        dp = p - q
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = p[0] * q[1] - p[1] * q[0]
        den = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(den) < 1e-15:
            return p
        return np.array([(n1 * dp[0] - n2 * dc[0]) / den, (n1 * dp[1] - n2 * dc[1]) / den])
    # orientation of the clipper
    sgn = 1.0 if (np.dot(clipper[:, 0], np.roll(clipper[:, 1], -1))
                  - np.dot(clipper[:, 1], np.roll(clipper[:, 0], -1))) > 0 else -1.0
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b, sgn):
                if not inside(s, a, b, sgn):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b, sgn):
                out.append(inter(s, e, a, b))
            s = e
    return np.array(out) if out else np.zeros((0, 2))


def bev_iou(a, b) -> float:
    """Rotated BEV IoU of two boxes given as (cx, cy, l, w, yaw)."""
    pa, pb = box_corners(*a), box_corners(*b)
    ia = _poly_area(_clip(pa, pb))
    ua = _poly_area(pa) + _poly_area(pb) - ia
    return float(ia / ua) if ua > 0 else 0.0


def duplicate_pairs(boxes: np.ndarray, thr: float = DUP_IOU) -> list:
    """Pairs (i, j, iou) of GT boxes [n, 5] = (cx, cy, l, w, yaw) with rotated BEV IoU > thr.
    Pairs whose centres are farther apart than the sum of their half-diagonals cannot overlap and are
    skipped exactly (not approximately)."""
    n = len(boxes)
    if n < 2:
        return []
    c = boxes[:, :2]
    r = 0.5 * np.hypot(boxes[:, 2], boxes[:, 3])
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    cand = np.argwhere(np.triu(d < (r[:, None] + r[None, :]), k=1))
    out = []
    for i, j in cand:
        v = bev_iou(boxes[i], boxes[j])
        if v > thr:
            out.append((int(i), int(j), float(v)))
    return out


def convex_hull(pts) -> np.ndarray:
    """Andrew's monotone chain: ``[n, 2]`` -> the hull ``[k, 2]`` counter-clockwise, no repeated end."""
    p = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


#: the camera-centred ANGLE image of the visibility pass: 0.1 deg per pixel, azimuth +-90, elevation +-45
ANG_RES_DEG, ANG_AZ_DEG, ANG_EL_DEG = 0.1, 90.0, 45.0
ANG_W, ANG_H = int(round(2 * ANG_AZ_DEG / ANG_RES_DEG)), int(round(2 * ANG_EL_DEG / ANG_RES_DEG))


def angle_poly(corners, cam_t):
    """``[8, 3]`` rig-frame cuboid corners -> its silhouette in the camera-centred ANGLE image (pixels:
    azimuth left->right, elevation top->bottom), or ``None`` when any corner is not in front of the
    camera centre (``x - t_x <= 0.1 m``). Occlusion along rays from ONE point does not depend on the
    lens model or the camera's rotation -- only on its position -- so this pass is independent of the
    projection under geometric review."""
    d = np.asarray(corners, dtype=np.float64) - np.asarray(cam_t, dtype=np.float64)[None]
    if (d[:, 0] <= 0.1).any():
        return None
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))                     # + = left
    el = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))  # + = up
    return convex_hull(np.c_[(ANG_AZ_DEG - az) / ANG_RES_DEG, (ANG_EL_DEG - el) / ANG_RES_DEG])


def visible_fractions(polys, depths, w: int = ANG_W, h: int = ANG_H) -> np.ndarray:
    """Painter's z-buffer: each polygon's visible share of its own (in-image) pixels, nearer polygons
    (smaller ``depths``) drawn last. ``None`` polygons -> NaN; a polygon with no in-image pixel -> NaN."""
    from PIL import Image, ImageDraw
    n = len(polys)
    out = np.full(n, np.nan)
    order = sorted((i for i in range(n) if polys[i] is not None and len(polys[i]) >= 3),
                   key=lambda i: -float(depths[i]))
    lab = Image.new("I", (w, h), 0)
    dl = ImageDraw.Draw(lab)
    own = {}
    for i in order:
        pts = [(float(x), float(y)) for x, y in polys[i]]
        dl.polygon(pts, fill=i + 1)
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        x0, y0 = max(int(math.floor(min(xs))), 0), max(int(math.floor(min(ys))), 0)
        x1, y1 = min(int(math.ceil(max(xs))) + 1, w), min(int(math.ceil(max(ys))) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in pts], fill=1)
        cnt = int(np.asarray(m, dtype=np.uint8).sum())
        if cnt:
            own[i] = (cnt, (x0, y0, x1, y1))
    L_ = np.asarray(lab)
    for i, (cnt, (x0, y0, x1, y1)) in own.items():
        out[i] = float((L_[y0:y1, x0:x1] == i + 1).sum()) / cnt
    return out


def bevb_px(x_m, y_m) -> tuple:
    """Metric (x fwd, y LEFT) -> BEV-B pixel; +y is drawn on the LEFT, x up."""
    return ((BEVB_Y - float(y_m)) * BEVB_PPM, (BEVB_X1 - float(x_m)) * BEVB_PPM)


def bevb_m(px, py) -> tuple:
    return (BEVB_X1 - float(py) / BEVB_PPM, BEVB_Y - float(px) / BEVB_PPM)


def target_region_polygon() -> np.ndarray:
    """The trainer's target region: |azimuth| <= 60 deg ∩ 0 <= x <= 60 ∩ |y| <= 16 (metres)."""
    xk = DECODE_Y_M / math.tan(math.radians(FOV_HALF_DEG))       # where the wedge meets |y| = 16
    return np.array([[0.0, 0.0], [xk, DECODE_Y_M], [DECODE_X_M, DECODE_Y_M],
                     [DECODE_X_M, -DECODE_Y_M], [xk, -DECODE_Y_M]])


def range_hist(d) -> dict:
    d = np.asarray(d, float)
    out = {}
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        out[f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f">={lo:g}m"] = int(((d >= lo) & (d < hi)).sum())
    return out


Z_EDGES = (-float("inf"), -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
           1.5, 2.0, float("inf"))


def hist_z(v) -> dict:
    """Counts of a height sample in fixed 0.25-0.5 m bins (metres), with n."""
    v = np.asarray(v, float)
    out = {"n": int(v.size)}
    for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        key = (f"<{hi:g}" if not np.isfinite(lo) else (f">={lo:g}" if not np.isfinite(hi) else
                                                      f"[{lo:g},{hi:g})"))
        out[key] = int(((v >= lo) & (v < hi)).sum())
    return out


def stats_summary(v) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


# ============================================================================================ #
# the run                                                                                        #
# ============================================================================================ #
def _p(*a):
    print(*a, flush=True)


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _by_path(name, p):
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):  # noqa: C901 -- one linear pipeline, sectioned
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="refcv6_gt_validation_step38000")
    ap.add_argument("--n-extra", type=int, default=2)
    ap.add_argument("--stills", default="9f8bedcfb9de:104,0191487845ef:64,34765c024267:108,X0:mid",
                    help="sha12:window(1-based) list; X0/X1 = the extra clips, 'mid' = middle window")
    ap.add_argument("--stills-only", action="store_true", help="render the stills, write them, stop")
    ap.add_argument("--visibility-only", action="store_true",
                    help="no frames: each GT target's visible fraction behind nearer GT cuboids, from the "
                         "camera centre (an angle-space z-buffer), -> <out-dir>/gt_visibility.json")
    ap.add_argument("--max-windows", type=int, default=0)
    # the coordinator's CPU-job rule (2026-09-26): START at >= 7.5 GB free on 3 consecutive samples
    # 30 s apart; ABORT if box-free RAM drops below 4.0 GB (the 8 GB rule is for GPU jobs)
    ap.add_argument("--start-ram-gb", type=float, default=7.5)
    ap.add_argument("--start-samples", type=int, default=3)
    ap.add_argument("--abort-ram-gb", type=float, default=4.0)
    ap.add_argument("--gate-wait-min", type=float, default=240.0)
    ap.add_argument("--agent-join-subset", default=None,
                    help="an eval-only LINE SUBSET of the config's --agent-join (byte-identical lines, "
                         "selected by clip_id); recorded as a departure with the extract record")
    ap.add_argument("--join3d-subset", default=None, help="the same for --join3d")
    ap.add_argument("--extract-record", default=None, help="extract_eval_joins.py's record(s), ';'-separated")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--title-s", type=float, default=2.5)
    ap.add_argument("--compact-mb", type=float, default=8.0)
    a = ap.parse_args(argv)
    t_all = time.time()
    import psutil

    def avail():
        return psutil.virtual_memory().available / 2 ** 30
    # ---- the RAM gate (CPU-job rule: >= start_ram_gb on start_samples consecutive samples) ------ #
    t0, ok_run, samples = time.time(), 0, []
    while ok_run < a.start_samples:
        av_ = avail()
        samples.append(round(av_, 2))
        ok_run = ok_run + 1 if av_ >= a.start_ram_gb else 0
        if ok_run >= a.start_samples:
            break
        if time.time() - t0 > a.gate_wait_min * 60:
            raise SystemExit(f"[gtval] BLOCKED: RAM floor {a.start_ram_gb} GB not held on "
                             f"{a.start_samples} consecutive samples in {a.gate_wait_min:.0f} min "
                             f"(last {samples[-5:]})")
        time.sleep(30)
    _p(f"[gtval] RAM gate passed: {a.start_samples} consecutive samples >= {a.start_ram_gb} GB "
       f"(last {samples[-a.start_samples:]}; waited {time.time() - t0:.0f} s)")
    os.environ["REFCV6_REPO"] = a.repo
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                   # no GPU is touched: CPU only
    out_dir, work = Path(a.out_dir), Path(a.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rec = {"tool": str(Path(__file__).resolve()), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "argv": sys.argv[1:], "model_forward": "NONE -- no network is built, loaded or run",
           "controls": {}, "departures": []}
    # ---- the code tree, asserted --------------------------------------------------------------- #
    sys.path.insert(0, a.battery_code)
    import refcv6_loader as L
    L.bootstrap()
    import tanitad
    want = os.path.normcase(os.path.abspath(os.path.join(a.repo, "stack")))
    got = os.path.normcase(os.path.abspath(tanitad.__file__))
    if not got.startswith(want):
        raise SystemExit(f"[gtval] tanitad imported from {got}, not from {want}")
    rec["tanitad_file"] = tanitad.__file__
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    tr = L.trainer()
    from tanitad.models import agent_slots as _as
    from tanitad.refs import refc_agents as _ra
    from tanitad.refs import refc_v3 as v3
    from tanitad.data import v2_dataset as v2d
    rmv = _by_path("rmv_for_gtval", TOOLS / "render_refcv6_map_video.py")
    rcv3 = _by_path("render_refcv3_video_for_gtval", Path(a.repo) / "taniteval/tools/render_refcv3_video.py")
    if tuple(_as.AGENT_CLASSES) != tuple(rmv.AGENT_CLASSES):
        raise SystemExit(f"[gtval] AGENT_CLASSES drifted: {_as.AGENT_CLASSES}")
    AGC = tuple(rmv.AGENT_CLASSES)
    PAL = tuple(rmv.AGENT_PALETTE)
    if abs(float(_ra.FOV_HALF_ANGLE_RAD) - math.radians(FOV_HALF_DEG)) > 1e-12:
        raise SystemExit("[gtval] refc_agents.FOV_HALF_ANGLE_RAD is not 60 deg")
    rng_ = _as.SlotDecodeRanges()
    if (float(rng_.x_fwd_m), float(rng_.y_half_m)) != (DECODE_X_M, DECODE_Y_M):
        raise SystemExit(f"[gtval] SlotDecodeRanges drifted: {rng_}")
    # ---- config + args + cfg: train()'s own path, WITHOUT the model ---------------------------- #
    cmd5 = md5_file(a.config)
    if cmd5 != a.expect_config_md5:
        raise SystemExit(f"[gtval] config md5 {cmd5} != {a.expect_config_md5}")
    config = L.load_config(a.config)
    targs, _argv, arec = L.parse_args(config)
    subst = {}
    for key, val in (("agent_join", a.agent_join_subset), ("join3d", a.join3d_subset)):
        if val:
            if not Path(val).exists():
                raise SystemExit(f"[gtval] --{key}-subset {val} does not exist")
            subst[key] = {"config_file": getattr(targs, key), "used_subset": val,
                          "subset_bytes": Path(val).stat().st_size}
            setattr(targs, key, val)
    if subst:
        ext = []
        for p_ in (a.extract_record or "").split(";"):
            if p_ and Path(p_).exists():
                ext.append(json.loads(Path(p_).read_text(encoding="utf-8")))
        if not ext or not all(e.get("pass") for e in ext):
            raise SystemExit("[gtval] a join subset is used but its extract record is missing or its "
                             "known-value control did not pass")
        rec["departures"].append({
            "what": "the joins are eval-only LINE SUBSETS of the kit's train+eval files (every kept line "
                    "byte-identical, selected by the line's own clip_id against the eval cache's clip set); "
                    "a full xz parse did not finish in 7 min under contention",
            "files": subst, "extract_records": ext})
    tr._read_anchor_artifact(targs)
    cfg = tr._pin_trainer_cfg(v3.refc_v3_smoke_config(targs.arm == "hier") if targs.smoke else
                              v3.refc_v3_sized_config(targs.size, hier=targs.arm == "hier"), targs)
    W = int(cfg.core.window)
    bank, cam_stamp = tr._build_rig_camera(cfg, targs)
    rec["config"] = {"path": a.config, "md5": cmd5, "window": W, "argv_remap": arec}
    rec["rig_camera"] = {"source": "refc_v3_train._build_rig_camera(cfg, args) -- the model's own "
                                   "per-clip camera bank, built from the run's config, no weights",
                         "stamp": str(cam_stamp)[:400]}
    _p(f"[gtval] trainer {tr.__file__}; window {W}; rig camera bank built")

    class GTWindows(tr.V3Dataset):
        """The trainer's dataset; only ``_agent_item`` and the episode providers are read."""
        u8_frames = True

    def clip_of(ep):
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]

    def sha12(cid):
        return hashlib.sha256(str(cid).encode("utf-8")).hexdigest()[:12]
    # ---- SELECTION: the 2 extra, LESS crowded clips (rule stated in RESULT.md BEFORE the run) --- #
    t_s = time.time()
    targs_sel = argparse.Namespace(**vars(targs))
    targs_sel.join3d = None
    targs_sel.map_gt_root = None
    s_ds, s_eps, s_rec = L.build_eval_dataset(None, cfg, targs_sel, config,
                                              with_perception_targets=True, dataset_cls=GTWindows)
    if s_rec.get("agent_join") is None:
        raise SystemExit("[gtval] the agent join was not attached")
    extr_path = getattr(targs, "agent_rig_extrinsics", None)
    extr_all = rcv3.load_extrinsics(extr_path, [clip_of(e) for e in s_eps])
    by_ep = {}
    for (e_i, t) in s_ds.index:
        by_ep.setdefault(e_i, []).append(int(t))
    fixed = [s.strip() for s in a.clips.split(",") if s.strip()]
    cands = []
    for e_i, ep in enumerate(s_eps):
        ts = sorted(by_ep.get(e_i, []))
        cid = clip_of(ep)
        n_lab = n_tg = 0
        for t in ts:
            it = s_ds._agent_item(ep, t + W - 1)
            if not bool(it["agent_label"]):
                continue
            n_lab += 1
            tv = _ra.visible_target_filter({"box": it["agent_box"][None], "valid": it["agent_valid"][None]})
            n_tg += int(tv["valid"].sum())
        cands.append({"sha12": sha12(cid), "_cid": cid, "episode_index": e_i, "n_windows": len(ts),
                      "n_windows_labelled": n_lab,
                      "targets_per_labelled_window": (n_tg / n_lab) if n_lab else None,
                      "has_extrinsics": cid in extr_all})
    elig = [c for c in cands if c["n_windows_labelled"] > 0 and c["has_extrinsics"]]
    med = float(np.median([c["targets_per_labelled_window"] for c in elig]))
    pool = sorted((c for c in elig if c["sha12"] not in fixed),
                  key=lambda c: (abs(c["targets_per_labelled_window"] - med), c["sha12"]))
    extra = pool[:a.n_extra]
    sel_rule = ("eligible = eval clips with >= 1 agent-labelled window and a per-clip extrinsics row; "
                "score = mean number of TRAINER TARGETS per labelled window (the 2-D GT through "
                "refc_agents.visible_target_filter at each eval window's NOW frame); take the "
                f"{a.n_extra} eligible clips (not among the 3 fixed) whose score is CLOSEST to the "
                "median score over all eligible clips; ties by sha12")
    sel = {"rule": sel_rule, "n_eval_clips": len(s_eps), "n_eligible": len(elig),
           "median_targets_per_labelled_window": med,
           "targets_per_labelled_window_quantiles": {
               q: float(np.quantile([c["targets_per_labelled_window"] for c in elig], q))
               for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "fixed": [{k: v for k, v in c.items() if not k.startswith("_")}
                     for c in cands if c["sha12"] in fixed],
           "extra": [{k: v for k, v in c.items() if not k.startswith("_")} for c in extra],
           "selection_s": round(time.time() - t_s, 1)}
    if len({c["sha12"] for c in cands if c["sha12"] in fixed}) != len(fixed):
        raise SystemExit(f"[gtval] a fixed clip is not in the eval cache: {fixed}")
    rmv.write_json(out_dir / "clip_selection_gt.json", sel)
    _p(f"[gtval] selection: median {med:.2f} targets/labelled window over {len(elig)} clips; extra "
       + ", ".join(f"{c['sha12']} ({c['targets_per_labelled_window']:.2f})" for c in extra)
       + f" [{sel['selection_s']} s]")
    run_list = [c for s in fixed for c in cands if c["sha12"] == s] + extra
    keep_cids = {c["_cid"] for c in run_list}
    del s_ds, s_eps
    # ---- the RENDER dataset: the chosen clips, WITH the 3-D join --------------------------------- #
    _orig_bvp, _orig_ctab = v2d.build_v2_providers, tr._clip_table_for_caches

    def _only(*aa, **kk):
        eps = _orig_bvp(*aa, **kk)
        kept = [e for e in eps if clip_of(e) in keep_cids]
        if len(kept) != len(keep_cids):
            raise SystemExit(f"[gtval] provider filter kept {len(kept)} of {len(keep_cids)}")
        return kept

    def _ctab(*aa, **kk):
        tab, ns = _orig_ctab(*aa, **kk)
        return {k: v for k, v in tab.items() if v in keep_cids}, ns
    targs_r = argparse.Namespace(**vars(targs))
    targs_r.map_gt_root = None
    v2d.build_v2_providers, tr._clip_table_for_caches = _only, _ctab
    try:
        e_ds, e_eps, drec = L.build_eval_dataset(None, cfg, targs_r, config,
                                                 with_perception_targets=True, dataset_cls=GTWindows)
    finally:
        v2d.build_v2_providers, tr._clip_table_for_caches = _orig_bvp, _orig_ctab
    if drec.get("agent_join") is None or drec.get("join3d") is None:
        raise SystemExit("[gtval] the agent join / 3-D join was not attached")
    rec["dataset"] = drec
    reader = e_ds.agent_join
    rec["provenance"] = provenance(targs, config, a)
    if subst:
        rec["provenance"]["join_subsets"] = subst
    # ---- the boxes video's frames, by (clip, window start) --------------------------------------- #
    bx_rows = [json.loads(ln) for ln in open(Path(a.boxes_raw) / "per_frame.jsonl", encoding="utf-8")
               if ln.strip()]
    bx_of = {(r["clip_sha12"], int(r["t_start_row"])): Path(a.boxes_frames) /
             f"c{int(r['clip_rank']) - 1}_w{int(r['win_rank']) - 1:04d}.png" for r in bx_rows}
    from PIL import Image, ImageDraw
    F = {"ban": rcv3.font(22, True), "hud": rcv3.font(16), "sub": rcv3.font(14),
         "subb": rcv3.font(14, True), "tiny": rcv3.font(12), "micro": rcv3.font(11),
         "med": rcv3.font(19, True), "title": rcv3.font(34, True), "lab": rcv3.font(12, True)}
    frames_dir = work / f"frames_{a.tag}"
    if frames_dir.exists() and not (a.stills_only or a.visibility_only):   # never wipe a finished reel's frames
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    stills_dir = out_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    ep_index_of = {clip_of(ep): i for i, ep in enumerate(e_eps)}
    wins_of = {}
    for (e_i, t) in e_ds.index:
        wins_of.setdefault(e_i, []).append(int(t))
    # stills: (sha12, window 1-based)
    still_req = []
    for tok in a.stills.split(","):
        s, w_ = tok.split(":")
        if s.startswith("X"):
            k = int(s[1:])
            if k >= len(extra):
                continue
            s = extra[k]["sha12"]
        still_req.append((s, w_))
    if a.visibility_only:
        return run_visibility(a, rec, run_list, fixed, e_ds, e_eps, ep_index_of, wins_of, W, extr_all, AGC,
                              out_dir, avail, rmv)
    pooled = new_acc()
    per_clip_out = []
    ctl = {"filter_mask_mismatch": 0, "n_windows_checked": 0, "proj_max_px": 0.0, "n_corners_checked": 0,
           "bev_roundtrip_max_px": 0.0, "n_bev_boxes_checked": 0, "self_iou_min": 1.0}
    stills_written = []
    t_r = time.time()
    clip_cache = {}
    for pass_ in (["stills"] if a.stills_only else ["stills", "full"]):
        stills_pass = pass_ == "stills"
        for ci, c in enumerate(run_list):
            cid, s12 = c["_cid"], c["sha12"]
            e_i = ep_index_of[cid]
            ep = e_eps[e_i]
            eid = int(ep.episode_id)
            wl = sorted(wins_of[e_i])
            if a.max_windows:
                wl = wl[:a.max_windows]
            # the camera: the model's own RigCamera, cross-checked against CylProjector on this clip
            rig_cam = bank.get(eid) if bank is not None else None
            pay_path = Path(ep.frames._cache.cache_dir) / ep.frames._cache.files[ep.frames._clip]
            pay = torch.load(str(pay_path), map_location="cpu", weights_only=False, mmap=True)
            fr_dict = dict(pay.get("frame") or {})
            del pay
            extr = extr_all.get(cid)
            proj = rcv3.CylProjector(fr_dict, extr)
            if rig_cam is None or extr is None:
                raise SystemExit(f"[gtval] {s12}: no RigCamera / extrinsics -- refusing to draw unprojected GT")
            cam_label = proj.label()
            acc = new_acc()
            if s12 not in clip_cache:
                # track numbers over ALL of the clip's windows, in time order, BEFORE any drawing: the
                # stills and the video then print the same "#n" for the same track
                tid_num: dict = {}
                for t_ in wl:
                    tl_ = reader.lookup_track_ids(eid, int(t_ + W - 1))   # an ARRAY (or None)
                    if tl_ is None:
                        continue
                    for tid in tl_:
                        tid_num.setdefault(str(tid), len(tid_num) + 1)
                clip_cache[s12] = tid_num
            tid_num = clip_cache[s12]
            want_idx = None
            if stills_pass:
                want_idx = {int(w_) - 1 if w_ != "mid" else len(wl) // 2 for s, w_ in still_req if s == s12}
                if not want_idx:
                    continue
            # title card
            _p(f"[clip {ci + 1}/{len(run_list)}] {s12} windows {len(wl)}; {cam_label}")
            for k_i, t in enumerate(wl):
                if want_idx is not None and k_i not in want_idx:
                    continue
                if avail() < a.abort_ram_gb:                       # the CPU-job abort floor
                    raise SystemExit(f"[gtval] ABORT: box-free RAM {avail():.2f} GB < "
                                     f"{a.abort_ram_gb} GB (CPU-job rule)")
                f_now = t + W - 1
                it = e_ds._agent_item(ep, f_now)
                lab = bool(it["agent_label"])
                valid = it["agent_valid"].numpy().astype(bool)
                box = it["agent_box"].double().numpy()
                yaw = it["agent_yaw"].double().numpy()
                cls = it["agent_cls"].numpy().astype(np.int64)
                occ = it["agent_occ"].double().numpy()
                cz = it["agent_cz"].double().numpy() if "agent_cz" in it else np.zeros(len(valid))
                hh = it["agent_h"].double().numpy() if "agent_h" in it else np.zeros(len(valid))
                zm = (it["agent_zh_mask"].numpy().astype(bool) if "agent_zh_mask" in it
                      else np.zeros(len(valid), bool))
                tids = reader.lookup_track_ids(eid, int(f_now)) if lab else None
                n_v = int(valid.sum())
                tnum = np.full(len(valid), -1, np.int64)
                if tids is not None:
                    if len(tids) != n_v:
                        raise SystemExit(f"[gtval] {s12} t{t}: {len(tids)} track ids for {n_v} rows")
                    for j, tid in enumerate(tids):
                        tnum[j] = tid_num[str(tid)]
                fr = filter_reasons(box[:, 0], box[:, 1], valid)
                tv = _ra.visible_target_filter({"box": it["agent_box"][None], "valid": it["agent_valid"][None]})
                mism = int((tv["valid"][0].numpy().astype(bool) != fr["keep"]).sum())
                if not stills_pass:
                    ctl["filter_mask_mismatch"] += mism
                    ctl["n_windows_checked"] += 1
                if mism:
                    raise SystemExit(f"[gtval] reason decomposition != visible_target_filter on {s12} t{t}")
                # per-box facts
                idx = np.nonzero(valid)[0]
                rng_m = np.hypot(box[idx, 0], box[idx, 1])
                names = [AGC[k] if 0 <= k < len(AGC) else "unknown" for k in cls[idx]]
                bottom = np.where(zm[idx], cz[idx] - hh[idx] / 2.0, np.nan)
                flags = []
                for jj, j in enumerate(idx):
                    fl = norm_flags(names[jj], box[j, 2], box[j, 3], hh[j] if zm[j] else None)
                    if fl:
                        flags.append((int(j), names[jj], fl, float(rng_m[jj])))
                dups = duplicate_pairs(np.c_[box[idx, :4], yaw[idx]])
                dups = [(int(idx[i]), int(idx[j]), v) for i, j, v in dups]
                accumulate(acc, lab, idx, names, box, hh, zm, bottom, rng_m, fr, flags, dups, occ, tnum,
                           int(it["agent_n_truncated"]))
                # ---- draw ------------------------------------------------------------------------ #
                rgb = ep.frames[f_now:f_now + 1][0][-3:].permute(1, 2, 0).contiguous().numpy()
                cam_raw = Image.fromarray(rgb)
                if cam_raw.size != (CAM_W, CAM_H):
                    raise SystemExit(f"[gtval] camera {cam_raw.size} != {(CAM_W, CAM_H)}")
                camB = cam_raw.copy()
                nlab, nlab_all, pxd, ncorn = draw_gt_camera(camB, rig_cam, proj, idx, names, box, yaw, cz,
                                                            hh, zm, fr, tnum, rng_m, PAL, AGC, F)
                ctl["proj_max_px"] = max(ctl["proj_max_px"], pxd)
                ctl["n_corners_checked"] += 0 if stills_pass else ncorn
                if pxd > 0.01:
                    raise SystemExit(f"[gtval] RigCamera vs CylProjector formula disagree by {pxd:.4f} px "
                                     f"on {s12} t{t} -- the GT would be drawn in the wrong place")
                bevB, rt = draw_gt_bev(idx, names, box, yaw, fr, PAL, AGC, F)
                ctl["bev_roundtrip_max_px"] = max(ctl["bev_roundtrip_max_px"], rt)
                ctl["n_bev_boxes_checked"] += 0 if stills_pass else len(idx)
                if len(idx):
                    j0 = idx[0]
                    ctl["self_iou_min"] = min(ctl["self_iou_min"],
                                              bev_iou((*box[j0, :4], yaw[j0]), (*box[j0, :4], yaw[j0])))
                src = bx_of.get((s12, int(t)))
                if src is not None and src.exists():
                    sv = Image.open(src).convert("RGB")
                    camA, bevA = sv.crop(CAMA_BOX), sv.crop(BEVA_BOX)
                    # camera A is a CROP of the boxes video's saved frame: its cuboids were drawn with the
                    # PRE-PATCH edge rule (both corners in frame); redrawing them needs the model forward
                    a_label = ("A \u2014 step-38,000 boxes video: MODEL boxes (3-D solid, agent dashed), GT "
                               "green \u00b7 drawn with the PRE-PATCH edge rule (GPU re-render pending)")
                else:
                    camA, bevA = cam_raw.copy(), None
                    a_label = ("A \u2014 RAW camera (this clip was not in the boxes video: no model render; "
                               "the GPU is held by A6)")
                t_lab = float(e_ds._now_s(ep, t)) if hasattr(e_ds, "_now_s") else None
                cv = compose(camA, a_label, camB, bevA, bevB, F, s12, ci, len(run_list), k_i, len(wl),
                             t_lab, cam_label, idx, names, fr, flags, dups, lab, zm, nlab, nlab_all, rng_m,
                             bottom, PAL, AGC)
                if stills_pass:
                    dst = stills_dir / f"still_{s12}_w{k_i + 1:03d}.png"
                    cv.save(dst)
                    stills_written.append(str(dst))
                    _p(f"[gtval] still {dst}")
                    continue
                fp = frames_dir / f"c{ci}_w{k_i:04d}.png"
                cv.save(fp, compress_level=3)
                if (k_i + 1) % 25 == 0:
                    _p(f"  w{k_i + 1:03d}/{len(wl)} gt {n_v} targets {int(fr['keep'].sum())} "
                       f"({time.time() - t_r:.0f} s, RAM {avail():.1f} GB)")
            if stills_pass:
                continue
            cs = finish_acc(acc)
            cs.update({"clip_sha12": s12, "role": "fixed (boxes video)" if s12 in fixed else "extra (median rule)",
                       "n_windows_rendered": len(wl), "camera": cam_label,
                       "n_tracks": len(tid_num)})
            per_clip_out.append(cs)
            rmv.write_json(out_dir / f"gt_stats_{s12}.json", {"provenance": rec["provenance"], **cs})
            draw_title(frames_dir / f"c{ci}_title.png", cs, ci, len(run_list), F, PAL, AGC)
            merge_acc(pooled, acc)
        if stills_pass:
            _p(f"[gtval] STILLS_READY {json.dumps(stills_written)}")
            (work / "STILLS_READY").write_text(json.dumps(stills_written), encoding="utf-8")
    rec["controls"] = ctl
    rec["stills"] = stills_written
    if a.stills_only:
        rmv.write_json(out_dir / "render_record_stills.json", rec)
        _p(f"[gtval] stills-only done in {time.time() - t_all:.0f} s; controls {json.dumps(ctl)}")
        return 0
    ps = finish_acc(pooled)
    ps.update({"clips": [c["clip_sha12"] for c in per_clip_out]})
    rmv.write_json(out_dir / "gt_stats_pooled.json", {"provenance": rec["provenance"], **ps})
    # ---- encode ------------------------------------------------------------------------------- #
    seq = work / f"seq_{a.tag}"
    if seq.exists():
        shutil.rmtree(seq)
    seq.mkdir(parents=True)
    n_title = max(1, int(round(a.title_s * a.fps)))
    order = []
    for ci in range(len(run_list)):
        order += [frames_dir / f"c{ci}_title.png"] * n_title
        order += sorted(frames_dir.glob(f"c{ci}_w*.png"))
    for n, src in enumerate(order):
        dst = seq / f"f{n:06d}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    ffmpeg = shutil.which("ffmpeg") or ("C:/Users/Admin/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_"
                                        "Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/"
                                        "ffmpeg.exe")

    def encode(dst, crf, scale=None):
        cmd = [ffmpeg, "-y", "-r", str(a.fps), "-i", str(seq / "f%06d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"]
        if scale:
            cmd[-4:-4] = ["-vf", f"scale={scale}:-2:flags=lanczos"]
        res = subprocess.run(cmd + [str(dst)], capture_output=True, text=True)
        if res.returncode != 0 or not Path(dst).exists() or Path(dst).stat().st_size == 0:
            raise SystemExit(f"[gtval] ffmpeg failed ({res.returncode}): {res.stderr[-800:]}")
        return {"path": str(dst), "bytes": Path(dst).stat().st_size, "crf": crf, "scale": scale}
    vid = {"frames": len(order), "fps": a.fps, "duration_s": round(len(order) / a.fps, 1),
           "full": encode(out_dir / f"{a.tag}.mp4", 20)}
    lim = a.compact_mb * 2 ** 20
    for crf in (30, 32, 34, 36, 38):
        vid["compact"] = encode(out_dir / f"{a.tag}_compact.mp4", crf, scale=1280)
        if vid["compact"]["bytes"] <= lim:
            break
    vid["compact_under_limit"] = vid["compact"]["bytes"] <= lim
    vpy = Path(a.repo) / "taniteval/tools/verify_mp4.py"
    vr = subprocess.run([sys.executable, str(vpy), vid["full"]["path"], vid["compact"]["path"]],
                        capture_output=True, text=True)
    vid["verify_mp4"] = {"exit": vr.returncode, "stdout_tail": vr.stdout[-2500:]}
    _p(vr.stdout[-2500:])
    rec["video"] = vid
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rmv.write_json(out_dir / "render_record.json", rec)
    _p(f"[gtval] DONE {vid['full']['path']} ({vid['full']['bytes']:,} B) + compact "
       f"{vid['compact']['bytes']:,} B; {len(order)} frames; controls {json.dumps(ctl)}; "
       f"{rec['wall_s']} s")
    return 0


# ============================================================================================ #
# the visibility pass (no frames)                                                                #
# ============================================================================================ #
VIS_RANGE = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 1e9))


def vis_bins(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    fin = v[np.isfinite(v)]
    n = max(int(fin.size), 1)
    return {"n": int(v.size), "n_assessable": int(fin.size),
            "n_not_assessable": int(v.size - fin.size),
            "n_fully_hidden": int((fin == 0).sum()), "n_lt_10pct_visible": int((fin < 0.1).sum()),
            "n_lt_50pct_visible": int((fin < 0.5).sum()), "n_ge_90pct_visible": int((fin >= 0.9).sum()),
            "frac_fully_hidden": float((fin == 0).sum()) / n, "frac_lt_10pct_visible": float((fin < 0.1).sum()) / n,
            "frac_lt_50pct_visible": float((fin < 0.5).sum()) / n,
            "frac_ge_90pct_visible": float((fin >= 0.9).sum()) / n,
            "median_visible": float(np.median(fin)) if fin.size else None}


def run_visibility(a, rec, run_list, fixed, e_ds, e_eps, ep_index_of, wins_of, W, extr_all, AGC, out_dir,
                   avail, rmv) -> int:
    """Each GT box's visible fraction behind NEARER GT boxes, seen from the per-clip camera CENTRE.

    Occluders are every valid GT row in front of the camera centre (targets AND filtered rows); the
    image is the camera-centred angle image (``angle_poly``), so neither the cylindrical model nor the
    camera rotation enters -- only the extrinsics' translation. Walls, trees, buildings and unlabelled
    objects are NOT in the GT, so the result is an UPPER bound on visibility."""
    t0 = time.time()
    per_clip, pooled_t, pooled_all = [], [], []
    pooled_cls, pooled_rng = {}, {}
    per_window_hidden = []
    for ci, c in enumerate(run_list):
        cid, s12 = c["_cid"], c["sha12"]
        ep = e_eps[ep_index_of[cid]]
        extr = extr_all.get(cid)
        if extr is None:
            raise SystemExit(f"[gtval:vis] {s12}: no extrinsics row -- the camera centre is unknown")
        cam_t = np.array([float(extr["x"]), float(extr["y"]), float(extr["z"])])
        wl = sorted(wins_of[ep_index_of[cid]])
        if a.max_windows:
            wl = wl[:a.max_windows]
        vt, va, by_cls, by_rng, hid_w = [], [], {}, {}, []
        for t in wl:
            if avail() < a.abort_ram_gb:
                raise SystemExit(f"[gtval:vis] ABORT: box-free RAM {avail():.2f} GB < {a.abort_ram_gb} GB")
            it = e_ds._agent_item(ep, t + W - 1)
            if not bool(it["agent_label"]):
                continue
            valid = it["agent_valid"].numpy().astype(bool)
            box = it["agent_box"].double().numpy()
            yaw = it["agent_yaw"].double().numpy()
            cls = it["agent_cls"].numpy().astype(np.int64)
            cz = it["agent_cz"].double().numpy() if "agent_cz" in it else np.zeros(len(valid))
            hh = it["agent_h"].double().numpy() if "agent_h" in it else np.zeros(len(valid))
            zm = (it["agent_zh_mask"].numpy().astype(bool) if "agent_zh_mask" in it
                  else np.zeros(len(valid), bool))
            idx = np.nonzero(valid)[0]
            fr = filter_reasons(box[:, 0], box[:, 1], valid)
            polys, depths = [], []
            for j in idx:
                cor = cuboid(box[j, 0], box[j, 1], cz[j] if zm[j] else 0.0, box[j, 2], box[j, 3],
                             hh[j] if zm[j] else 0.0, yaw[j])
                polys.append(angle_poly(cor, cam_t))
                depths.append(float(np.linalg.norm(cor.mean(0) - cam_t)))
            vf = visible_fractions(polys, depths)
            n_hid = 0
            for jj, j in enumerate(idx):
                va.append(vf[jj])
                if not fr["keep"][j]:
                    continue
                vt.append(vf[jj])
                nm = AGC[cls[j]] if 0 <= cls[j] < len(AGC) else "unknown"
                by_cls.setdefault(nm, []).append(vf[jj])
                r_ = float(math.hypot(box[j, 0], box[j, 1]))
                rk = next(f"{lo:g}-{hi:g}m" if hi < 1e8 else f">={lo:g}m" for lo, hi in VIS_RANGE if lo <= r_ < hi)
                by_rng.setdefault(rk, []).append(vf[jj])
                n_hid += int(np.isfinite(vf[jj]) and vf[jj] < 0.1)
            hid_w.append(n_hid)
        cs = {"clip_sha12": s12, "role": "fixed (boxes video)" if s12 in fixed else "extra (median rule)",
              "n_windows": len(hid_w), "camera_centre_rig_m": [round(float(x), 3) for x in cam_t],
              "targets": vis_bins(vt), "all_gt_rows": vis_bins(va),
              "targets_by_class": {k: vis_bins(v) for k, v in sorted(by_cls.items())},
              "targets_by_range": {k: vis_bins(v) for k, v in by_rng.items()},
              "targets_lt_10pct_visible_per_window": stats_summary(hid_w)}
        per_clip.append(cs)
        pooled_t += vt
        pooled_all += va
        per_window_hidden += hid_w
        for k, v in by_cls.items():
            pooled_cls.setdefault(k, []).extend(v)
        for k, v in by_rng.items():
            pooled_rng.setdefault(k, []).extend(v)
        _p(f"[gtval:vis] {s12}: targets {cs['targets']['n']:,}, <10 % visible "
           f"{cs['targets']['frac_lt_10pct_visible'] * 100:.1f} %, fully hidden "
           f"{cs['targets']['frac_fully_hidden'] * 100:.1f} %, median visible {cs['targets']['median_visible']}")
    out = {"what": "each GT target's visible fraction behind nearer GT cuboids, from the per-clip camera "
                   "centre (angle-space z-buffer, 0.1 deg px); an UPPER bound on visibility (static "
                   "occluders are not in the GT); independent of the lens model and camera rotation",
           "stamp": "refcv6 eval set, the trainer's per-window GT (V3Dataset._agent_item) and target filter",
           "method": {"angle_res_deg": ANG_RES_DEG, "az_half_deg": ANG_AZ_DEG, "el_half_deg": ANG_EL_DEG,
                      "occluders": "every valid GT row in front of the camera centre (targets + filtered)",
                      "order": "painter's, by camera-centre distance to the box centre",
                      "not_assessable": "a box with any corner <= 0.1 m in front of the camera centre, "
                                        "or with no pixel in the angle image"},
           "pooled": {"n_windows": len(per_window_hidden), "targets": vis_bins(pooled_t),
                      "all_gt_rows": vis_bins(pooled_all),
                      "targets_by_class": {k: vis_bins(v) for k, v in sorted(pooled_cls.items())},
                      "targets_by_range": {k: vis_bins(v) for k, v in pooled_rng.items()},
                      "targets_lt_10pct_visible_per_window": stats_summary(per_window_hidden)},
           "clips": per_clip, "provenance": rec.get("provenance"), "departures": rec.get("departures"),
           "wall_s": round(time.time() - t0, 1)}
    rmv.write_json(out_dir / "gt_visibility.json", out)
    p = out["pooled"]["targets"]
    _p(f"[gtval:vis] DONE pooled targets {p['n']:,}: fully hidden {p['frac_fully_hidden'] * 100:.1f} %, "
       f"<10 % visible {p['frac_lt_10pct_visible'] * 100:.1f} %, <50 % {p['frac_lt_50pct_visible'] * 100:.1f} %, "
       f">=90 % {p['frac_ge_90pct_visible'] * 100:.1f} %; not assessable {p['n_not_assessable']}; "
       f"{out['wall_s']} s")
    return 0


# ============================================================================================ #
# accumulation                                                                                   #
# ============================================================================================ #
def new_acc() -> dict:
    return {"n_windows": 0, "n_windows_no_label": 0, "n_windows_labelled_clear": 0, "per_window_gt": [],
            "per_window_targets": [], "cls_gt": {}, "cls_target": {}, "range_all": [], "range_target": [],
            "n_outside_fov": 0, "n_valid": 0, "reason_any": [0, 0, 0, 0], "reason_primary": [0, 0, 0, 0],
            "sizes": {}, "flags": [], "n_flags": 0, "bottom": {}, "bottom_by_range": {},
            "n_float": 0, "n_zh": 0, "dups": [], "n_dups": 0, "n_dups_targets": 0, "n_unknown_cls": 0,
            "occ": {}, "n_truncated": 0, "n_dup_track_in_frame": 0, "cz": {}}


def accumulate(acc, lab, idx, names, box, hh, zm, bottom, rng_m, fr, flags, dups, occ, tnum, n_trunc):
    acc["n_windows"] += 1
    if not lab:
        acc["n_windows_no_label"] += 1
        return
    if len(idx) == 0:
        acc["n_windows_labelled_clear"] += 1
    keep = fr["keep"]
    acc["per_window_gt"].append(len(idx))
    acc["per_window_targets"].append(int(keep.sum()))
    acc["n_valid"] += len(idx)
    acc["n_truncated"] += int(n_trunc)
    fov_out = np.arctan2(np.abs(box[idx, 1]), box[idx, 0]) > math.radians(FOV_HALF_DEG)
    acc["n_outside_fov"] += int(fov_out.sum())
    for k in range(4):
        acc["reason_any"][k] += int(fr["reasons"][k][idx].sum())
        acc["reason_primary"][k] += int((fr["primary"][idx] == k).sum())
    for jj, j in enumerate(idx):
        nm = names[jj]
        acc["cls_gt"][nm] = acc["cls_gt"].get(nm, 0) + 1
        if keep[j]:
            acc["cls_target"][nm] = acc["cls_target"].get(nm, 0) + 1
        acc["range_all"].append(float(rng_m[jj]))
        if keep[j]:
            acc["range_target"].append(float(rng_m[jj]))
        s = acc["sizes"].setdefault(nm, {"L": [], "W": [], "H": []})
        s["L"].append(float(box[j, 2]))
        s["W"].append(float(box[j, 3]))
        if zm[j]:
            s["H"].append(float(hh[j]))
            acc["n_zh"] += 1
            b = float(bottom[jj])
            acc["bottom"].setdefault(nm, []).append(b)
            # the raw join3d centre z as well: if the file's `cz` were ALREADY a base, the true bottom
            # would be `cz` itself -- both readings are banked so the geometric review can decide
            acc.setdefault("cz", {}).setdefault(nm, []).append(float(bottom[jj] + hh[j] / 2.0))
            rb = "0-20m" if rng_m[jj] < 20 else ("20-40m" if rng_m[jj] < 40 else ">=40m")
            acc["bottom_by_range"].setdefault(rb, []).append(b)
            if abs(b) > FLOAT_M:
                acc["n_float"] += 1
        if nm == "unknown":
            acc["n_unknown_cls"] += 1
        o = str(int(occ[j])) if np.isfinite(occ[j]) else "nan"
        acc["occ"][o] = acc["occ"].get(o, 0) + 1
    tn = tnum[idx]
    tn = tn[tn > 0]
    acc["n_dup_track_in_frame"] += int(len(tn) - len(set(tn.tolist())))
    acc["n_flags"] += len(flags)
    for f in flags[:3]:
        if len(acc["flags"]) < 60:
            acc["flags"].append({"class": f[1], "flags": f[2], "range_m": round(f[3], 1)})
    acc["n_dups"] += len(dups)
    acc["n_dups_targets"] += sum(1 for i, j, _ in dups if keep[i] and keep[j])
    for d in dups[:2]:
        if len(acc["dups"]) < 40:
            acc["dups"].append({"iou": round(d[2], 3), "classes": [names[list(idx).index(d[0])],
                                                                   names[list(idx).index(d[1])]]})


def merge_acc(dst, src):
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] += v
        elif isinstance(v, list) and k in ("reason_any", "reason_primary"):
            for i in range(4):
                dst[k][i] += v[i]
        elif isinstance(v, list):
            dst[k] += v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    t = dst[k].setdefault(kk, {"L": [], "W": [], "H": []})
                    for q in ("L", "W", "H"):
                        t[q] += vv[q]
                elif isinstance(vv, list):
                    dst[k].setdefault(kk, [])
                    dst[k][kk] += vv
                else:
                    dst[k][kk] = dst[k].get(kk, 0) + vv


def finish_acc(acc) -> dict:
    nv = max(acc["n_valid"], 1)
    out = {
        "n_windows": acc["n_windows"], "n_windows_no_label": acc["n_windows_no_label"],
        "n_windows_labelled_clear": acc["n_windows_labelled_clear"],
        "n_gt_boxes": acc["n_valid"],
        "gt_per_labelled_window": stats_summary(acc["per_window_gt"]),
        "targets_per_labelled_window": stats_summary(acc["per_window_targets"]),
        "gt_by_class": dict(sorted(acc["cls_gt"].items(), key=lambda kv: -kv[1])),
        "targets_by_class": dict(sorted(acc["cls_target"].items(), key=lambda kv: -kv[1])),
        "gt_by_class_per_labelled_window": {k: v / max(len(acc["per_window_gt"]), 1)
                                            for k, v in acc["cls_gt"].items()},
        "range_hist_all_gt": range_hist(acc["range_all"]),
        "range_hist_targets": range_hist(acc["range_target"]),
        "frac_outside_120deg_fov": acc["n_outside_fov"] / nv,
        "filter": {"n_gt": acc["n_valid"], "n_kept_as_targets": int(sum(acc["per_window_targets"])),
                   "frac_removed": 1.0 - sum(acc["per_window_targets"]) / nv,
                   "by_reason_any": {r: {"n": acc["reason_any"][k], "frac_of_gt": acc["reason_any"][k] / nv}
                                     for k, r in enumerate(REASONS)},
                   "by_primary_reason": {r: {"n": acc["reason_primary"][k],
                                             "frac_of_gt": acc["reason_primary"][k] / nv}
                                         for k, r in enumerate(REASONS)},
                   "rule": "refc_agents.visible_target_filter (refc_agents.py:410-433): valid & "
                           "|atan2(cy,cx)| <= 60 deg & 0 <= cx <= 60 & |cy| <= 16"},
        "sizes_per_class": {k: {q: stats_summary(v[q]) for q in ("L", "W", "H")}
                            for k, v in sorted(acc["sizes"].items())},
        "size_outliers_vs_literal_norms": {"n": acc["n_flags"], "frac_of_gt": acc["n_flags"] / nv,
                                           "norms_m": NORMS, "examples": acc["flags"][:40]},
        "box_bottom_cz_minus_h_over_2": {
            "n_with_3d_label": acc["n_zh"], "frac_gt_with_3d_label": acc["n_zh"] / nv,
            "frame": "the rig frame of obstacle.offline (reference_frame 'rig', agent_cuboid_gt.py "
                     "module docstring): +x fwd, +y left, +z up; bottom = join3d cz - h/2 "
                     "(agent_cuboid_gt.base_from_centre :430); the road plane is z = 0 only if the rig "
                     "origin sits on the road -- a systematic offset shows up in these medians",
            "per_class": {k: stats_summary(v) for k, v in sorted(acc["bottom"].items())},
            "per_class_hist": {k: hist_z(v) for k, v in sorted(acc["bottom"].items())},
            "pooled_hist": hist_z([b for v in acc["bottom"].values() for b in v]),
            "raw_join3d_cz_per_class": {k: stats_summary(v) for k, v in sorted(acc.get("cz", {}).items())},
            "by_range": {k: stats_summary(v) for k, v in sorted(acc["bottom_by_range"].items())},
            "n_abs_gt_0p5m": acc["n_float"], "frac_of_3d_labelled": acc["n_float"] / max(acc["n_zh"], 1),
            "ground": "rig frame z = 0 (box3d_head.ground_standing_z: bottom median 0.039 m automobile, "
                      "MEASURED 2026-09-16 on 51,844 eval cuboids)"},
        "duplicates_bev_iou_gt_0p5": {"n_pairs": acc["n_dups"], "n_pairs_both_targets": acc["n_dups_targets"],
                                      "examples": acc["dups"][:20]},
        "n_unknown_class": acc["n_unknown_cls"], "occ_flag_counts": acc["occ"],
        "n_rows_truncated_by_pad": acc["n_truncated"],
        "n_duplicate_track_ids_within_a_frame": acc["n_dup_track_in_frame"],
    }
    return out


def provenance(targs, config, a) -> dict:
    aj, j3 = getattr(targs, "agent_join", None), getattr(targs, "join3d", None)
    return {
        "published_source": "PhysicalAI-AV labels/obstacle.offline cuboids (rig frame; the B1 eval "
                            "mirror obstacle_offline_b1eval)",
        "join_2d": {"file": aj, "bytes": Path(aj).stat().st_size if aj and Path(aj).exists() else None,
                    "built_by": "stack/scripts/build_b1_agent_join.py (join_clip_raw :347; per-track "
                                "nearest sample world_agents_at + planar SE(2) ego_frame_agents :409-410)",
                    "read_by": "stack/scripts/train_p8_occupancy.py JoinFileReader (:240; lookup :465, "
                               "lookup_classes :472, lookup_track_ids :481)"},
        "join_3d": {"file": j3, "bytes": Path(j3).stat().st_size if j3 and Path(j3).exists() else None,
                    "built_by": "stack/scripts/build_b1_agent_join_3d.py (center_z / size_z re-attached "
                                "BY TRACK ID; agent_cuboid_gt.py module docstring)",
                    "read_by": "stack/tanitad/data/agent_cuboid_gt.py AgentJoin3D (:298), zh_for_frame "
                               "(:411), base_from_centre (:430)"},
        "per_window_gt": "stack/scripts/refc_v3_train.py:2764-2864 V3Dataset._agent_item (raw rows, NO "
                         "visibility filter; agent_slots.targets_from_join :1014-1077)",
        "trainer_target_filter": "stack/tanitad/refs/refc_agents.py:410-433 visible_target_filter -> "
                                 "filter_targets_to_visible :336-406; decode box agent_slots.py:526-539 "
                                 "SlotDecodeRanges (x 60 m, |y| 16 m); applied in box3d_set_loss "
                                 "(box3d_head.py:376-411) and agent_losses (refc_agents.py:849-852)",
        "pad": {"agent_pad": ((config.get("agent_join_stats") or {}).get("train") or {}).get("agent_pad"),
                "source": "config.json agent_join_stats.train.agent_pad"},
        "tree": a.repo,
    }


# ============================================================================================ #
# drawing                                                                                        #
# ============================================================================================ #
def _dashed_line(d, p, q, fill, width=1, dash=6.0, gap=4.0):
    x0, y0 = p
    x1, y1 = q
    seg = math.hypot(x1 - x0, y1 - y0)
    if seg <= 0:
        return
    ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
    s = 0.0
    while s < seg:
        e = min(s + dash, seg)
        d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
        s = e + gap


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))

_RMV_EDGE_RUNS = None


def _edge_runs(rig_cam, cor):
    """``render_refcv6_map_video.cuboid_edge_runs`` -- ONE implementation of the sampled, border-clipped
    cuboid edges, loaded by path from this tool's own directory and reused (box-head audit 2026-09-27: the
    old both-corners-in-frame rule turned border-crossing cuboids into single flat faces)."""
    global _RMV_EDGE_RUNS
    if _RMV_EDGE_RUNS is None:
        _RMV_EDGE_RUNS = _by_path("rmv_edge_runs_for_gtval", TOOLS / "render_refcv6_map_video.py").cuboid_edge_runs
    return _RMV_EDGE_RUNS(rig_cam, cor)


def _dashed_polyline(d, pts, fill, width=1, dash=6.0, gap=4.0):
    """A dashed polyline whose dash phase CONTINUES across vertices (a sampled edge has many short parts,
    and restarting the pattern on each part would draw it solid)."""
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        s = 0.0
        while seg > 0 and s < seg:
            e = min(s + left, seg)
            if on:
                ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
                d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
            left -= e - s
            s = e
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def cyl3(proj, p3):
    """``render_refcv3_video.CylProjector.__call__``'s formula for a 3-D rig-frame point (the class
    itself takes ground points only): vehicle -> camera ``R.T @ (p - t)``, then cylindrical
    ``u = cx + f atan2(x, z)``, ``v = cy + f y / hypot(x, z)``. None where the class would drop it."""
    pc = proj.R.T @ (np.asarray(p3, dtype=np.float64) - proj.t)
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    if proj.proj != "cylindrical":
        return None
    rho = math.hypot(x, z)
    if rho < 0.5 or z <= 0.0:
        return None
    return (proj.cx + proj.f * math.atan2(x, z), proj.cy + proj.f * y / rho)


def cuboid(cx, cy, cz, l, w, h, yaw) -> np.ndarray:
    fp = box_corners(cx, cy, l, w, yaw)
    z0, z1 = cz - h / 2.0, cz + h / 2.0
    return np.r_[np.c_[fp, np.full(4, z0)], np.c_[fp, np.full(4, z1)]]


def draw_gt_camera(cam, rig_cam, proj, idx, names, box, yaw, cz, hh, zm, fr, tnum, rng_m, PAL, AGC, F):
    """GT ONLY. Targets solid (width 2), filtered dashed (width 1) + "filtered: <reason>"; a row without
    a 3-D label is its ground footprint (z 0, h 0), never an invented height. Labels are placed
    nearest-first and skipped where they would overlap one already placed (count returned)."""
    import torch
    from PIL import ImageDraw
    d = ImageDraw.Draw(cam, "RGBA")
    order = sorted(range(len(idx)), key=lambda jj: -rng_m[jj])      # far first, near drawn on top
    placed, labels = [], []
    max_px, n_corn = 0.0, 0
    for jj in order:
        j = idx[jj]
        nm = names[jj]
        col = PAL[AGC.index(nm)] if nm in AGC else (200, 200, 200)
        has_z = bool(zm[j])
        cor = cuboid(box[j, 0], box[j, 1], cz[j] if has_z else 0.0, box[j, 2], box[j, 3],
                     hh[j] if has_z else 0.0, yaw[j])
        c_, r_, ok = rig_cam.project(torch.as_tensor(cor, dtype=torch.float64))
        c_, r_, ok = c_.numpy(), r_.numpy(), ok.numpy().astype(bool)
        runs = _edge_runs(rig_cam, cor)
        if not runs:
            # ⭐ decided on the EDGES, not the corners: a near cuboid can cross the whole frame with all
            # eight corners outside it (a bus alongside), and the corner rule skipped it entirely
            continue
        # the control: an INDEPENDENT implementation (render_refcv3_video.CylProjector's formula and
        # its per-clip R, t, f, cx, cy, lifted to 3-D) must land every drawn corner on the same pixel
        for k in np.nonzero(ok)[0]:
            qq = cyl3(proj, cor[k])
            if qq is not None:
                max_px = max(max_px, math.hypot(qq[0] - float(c_[k]), qq[1] - float(r_[k])))
                n_corn += 1
        kept = bool(fr["keep"][j])
        for _e, run in runs:
            if kept:
                d.line(run, fill=col + (255,), width=2, joint="curve")
            else:
                _dashed_polyline(d, run, col + (200,), width=1)
        top = cor[4:].mean(0) if has_z else cor[:4].mean(0)
        tc, trow, tok = rig_cam.project(torch.as_tensor(top[None], dtype=torch.float64))
        if not bool(tok[0]):
            continue
        tag = f"{nm} {rng_m[jj]:.0f}m" + (f" #{tnum[j]}" if tnum[j] > 0 else "") + ("" if has_z else " (2-D)")
        if not kept:
            tag += f" \u00b7 filtered: {REASON_SHORT[int(fr['primary'][j])]}"
        labels.append((float(tc[0]), float(trow[0]), tag, col, kept))
    n_all = len(labels)
    n_shown = 0
    for x, y, tag, col, kept in reversed(labels):                     # nearest first
        tw = d.textlength(tag, font=F["micro"]) + 6
        x0, y0 = min(max(2, x - tw / 2), CAM_W - tw - 2), max(22, y - 16)
        rect = (x0, y0, x0 + tw, y0 + 14)
        if any(not (rect[2] < q[0] or rect[0] > q[2] or rect[3] < q[1] or rect[1] > q[3]) for q in placed):
            continue
        placed.append(rect)
        d.rectangle(rect, fill=(8, 10, 14, 190))
        d.text((x0 + 3, y0 + 1), tag, fill=col if kept else (200, 200, 200), font=F["micro"])
        n_shown += 1
    return n_shown, n_all, max_px, n_corn


def draw_gt_bev(idx, names, box, yaw, fr, PAL, AGC, F):
    """GT ONLY, the WIDE field. Returns (image, the max BEV round-trip error in px)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (BEVB_W, BEVB_H), C_PANEL)
    d = ImageDraw.Draw(im, "RGBA")
    for xm in range(int(BEVB_X0), int(BEVB_X1) + 1, 10):
        py = bevb_px(xm, 0)[1]
        d.line([(0, py), (BEVB_W, py)], fill=(40, 48, 60), width=1)
        d.text((3, py - 13), f"{xm} m", fill=C_DIM, font=F["micro"])
    for ym in range(-30, 31, 10):
        px = bevb_px(0, ym)[0]
        d.line([(px, 0), (px, BEVB_H)], fill=(34, 40, 52), width=1)
    reg = [bevb_px(x, y) for x, y in target_region_polygon()]
    d.polygon(reg, fill=(110, 231, 138, 22))
    for p, q in zip(reg, reg[1:] + reg[:1]):
        _dashed_line(d, p, q, C_GT + (170,), width=1)
    ex, ey = bevb_px(0, 0)
    d.polygon([(ex, ey - 9), (ex - 5, ey + 4), (ex + 5, ey + 4)], fill=(255, 255, 255))
    rt = 0.0
    tags = []
    for jj, j in enumerate(idx):
        nm = names[jj]
        col = PAL[AGC.index(nm)] if nm in AGC else (200, 200, 200)
        cor = box_corners(box[j, 0], box[j, 1], box[j, 2], box[j, 3], yaw[j])
        pts = [bevb_px(x, y) for x, y in cor]
        for (x, y), (px_, py_) in zip(cor, pts):                     # the control: pixel -> metres
            xb, yb = bevb_m(px_, py_)
            rt = max(rt, math.hypot(xb - x, yb - y) * BEVB_PPM)
        if bool(fr["keep"][j]):
            d.polygon(pts, fill=col + (70,), outline=col + (255,))
        else:
            for p, q in zip(pts, pts[1:] + pts[:1]):
                _dashed_line(d, p, q, col + (190,), width=1, dash=4.0, gap=3.0)
        cx_, cy_ = bevb_px(box[j, 0], box[j, 1])
        if 0 <= cx_ < BEVB_W and 0 <= cy_ < BEVB_H:
            tags.append((cx_, cy_, CLASS_TAG.get(nm, "?"), col))
    placed = []
    for cx_, cy_, tg, col in tags:
        tw = d.textlength(tg, font=F["micro"])
        rect = (cx_ - tw / 2 - 1, cy_ - 6, cx_ + tw / 2 + 1, cy_ + 6)
        if any(not (rect[2] < q[0] or rect[0] > q[2] or rect[3] < q[1] or rect[1] > q[3]) for q in placed):
            continue
        placed.append(rect)
        d.text((rect[0] + 1, rect[1] - 1), tg, fill=col, font=F["micro"], stroke_width=2,
               stroke_fill=(8, 10, 14))
    return im, rt


def compose(camA, a_label, camB, bevA, bevB, F, s12, ci, n_clips, k_i, n_w, t_lab, cam_label, idx,
            names, fr, flags, dups, lab, zm, nlab, nlab_all, rng_m, bottom, PAL, AGC):
    from PIL import Image, ImageDraw
    cv = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(cv)
    d.rectangle([0, 0, W_TOT, H_BAN], fill=C_BAN)
    d.text((PAD, 6), "refcv6 agent GROUND TRUTH \u2014 VALIDATION \u00b7 camera B + BEV B show ONLY the GT "
                     "(no model output)", fill=C_FG, font=F["ban"])
    d.text((PAD, 40), "GT = agent join (obstacle.offline) + join3d, the trainer's per-window block \u00b7 "
                      "SOLID = a trainer TARGET (trained on) \u00b7 DASHED = filtered by "
                      "visible_target_filter (120\u00b0 field \u2229 x 0\u202660 m, |y| \u2264 16 m)",
           fill=C_WARN, font=F["hud"])
    rt = (f"clip {ci + 1}/{n_clips} \u00b7 sha12 {s12} \u00b7 window {k_i + 1}/{n_w}"
          + ("" if t_lab is None else f" \u00b7 t {t_lab:.1f} s"))
    d.text((W_TOT - PAD - d.textlength(rt, font=F["med"]), 8), rt, fill=C_FG, font=F["med"])
    cv.paste(camA, (PAD, Y_CAMA))
    d.rectangle([PAD, Y_CAMA, PAD + CAM_W, Y_CAMA + 20], fill=(9, 12, 17))
    d.text((PAD + 6, Y_CAMA + 2), a_label, fill=C_DIM, font=F["tiny"])
    cv.paste(camB, (PAD, Y_CAMB))
    d.rectangle([PAD, Y_CAMB, PAD + CAM_W, Y_CAMB + 20], fill=(9, 12, 17))
    d.text((PAD + 6, Y_CAMB + 2),
           f"B \u2014 GT ONLY: cuboids in class colours \u00b7 label = class \u00b7 range \u00b7 track # "
           f"\u00b7 {nlab}/{nlab_all} labels shown (overlaps skipped) \u00b7 {cam_label}"[:160],
           fill=C_FG, font=F["tiny"])
    # the geometric-review caption, on EVERY camera image (Master Mind, 2026-09-27 00:25)
    for y_top in (Y_CAMA, Y_CAMB):
        yb = y_top + CAM_H - 18
        d.rectangle([PAD, yb, PAD + CAM_W, y_top + CAM_H], fill=(40, 22, 8))
        d.text((PAD + 6, yb + 2), REVIEW_CAPTION, fill=C_WARN, font=F["lab"])
    # right column: BEV A (crop) and BEV B
    xa = X_R
    if bevA is not None:
        cv.paste(bevA, (xa, Y_CAMA))
    else:
        d.rectangle([xa, Y_CAMA, xa + 366, Y_CAMA + 663], outline=(40, 48, 60))
        d.text((xa + 12, Y_CAMA + 300), "BEV A: no model render\nfor this clip", fill=C_DIM, font=F["sub"])
    xb = xa + 366 + 14
    d.text((xb, Y_CAMA + 2), "BEV B \u2014 GT ONLY \u00b7 judge GT placement HERE", fill=C_GT, font=F["subb"])
    d.text((xb, Y_CAMA + 22), "no camera model \u00b7 x \u221220\u202680 m, y \u00b132 m \u00b7 green = "
                              "target region", fill=C_DIM, font=F["tiny"])
    cv.paste(bevB, (xb, Y_CAMA + 44))
    d.rectangle([xb - 3, Y_CAMA + 41, xb + BEVB_W + 2, Y_CAMA + 44 + BEVB_H + 2], outline=C_GT, width=3)
    # the legend (right column, below the BEVs)
    yl = Y_CAMA + (BEVA_BOX[3] - BEVA_BOX[1]) + 14
    d.text((X_R, yl), "GT CLASS COLOURS (camera B, BEV B) \u00b7 tag", fill=C_FG, font=F["subb"])
    yl += 20
    for n_, nm in enumerate(AGC):
        cx0, cy0 = X_R + (n_ % 3) * 280, yl + (n_ // 3) * 17
        d.rectangle([cx0, cy0 + 3, cx0 + 10, cy0 + 13], fill=PAL[n_])
        d.text((cx0 + 15, cy0), f"{nm} \u00b7 {CLASS_TAG[nm]}", fill=C_FG, font=F["tiny"])
    yl += 17 * 4 + 6
    for ln in ("SOLID = a trainer TARGET: inside the 120\u00b0 field AND the decode box (x 0\u202660 m, "
               "|y| \u2264 16 m)",
               "DASHED + 'filtered: <reason>' = GT the trainer's visible_target_filter removes (never "
               "trained on)",
               "'(2-D)' = no 3-D label for this row: drawn as its ground footprint, no invented height",
               "#n = the join's track_id, renumbered per clip (raw ids are never printed) \u00b7 range = "
               "distance from the rig origin",
               "camera A / BEV A: the model's boxes from the step-38,000 video (3-D head solid, agent "
               "head dashed); GT there is green"):
        d.text((X_R, yl), ln, fill=C_DIM, font=F["tiny"])
        yl += 17
    # the census (left column, below camera B)
    y = Y_CAMB + CAM_H + 10
    xs = PAD
    keep = fr["keep"]
    n_t = int(keep[idx].sum()) if len(idx) else 0
    if not lab:
        d.text((xs, y), "NO_LABEL window: the join has no row for this frame (not an empty road)",
               fill=C_BAD, font=F["subb"])
        y += 22
    d.text((xs, y), f"GT this frame: {len(idx)} boxes \u00b7 {n_t} trainer targets \u00b7 {len(idx) - n_t} "
                    f"filtered", fill=C_FG, font=F["subb"])
    y += 20
    rs = " \u00b7 ".join(f"{REASON_SHORT[k]} {int((fr['primary'][idx] == k).sum())}" for k in range(4))
    d.text((xs, y), f"filtered by first reason: {rs}", fill=C_DIM, font=F["tiny"])
    y += 18
    counts = {}
    for jj, j in enumerate(idx):
        t_ = counts.setdefault(names[jj], [0, 0])
        t_[0 if keep[j] else 1] += 1
    col_x = [xs, xs + 340, xs + 680]
    for n_, (nm, (kt, kf)) in enumerate(sorted(counts.items(), key=lambda kv: -sum(kv[1]))[:9]):
        cx0 = col_x[n_ % 3]
        cy0 = y + (n_ // 3) * 17
        colr = PAL[AGC.index(nm)] if nm in AGC else (200, 200, 200)
        d.rectangle([cx0, cy0 + 3, cx0 + 10, cy0 + 13], fill=colr)
        d.text((cx0 + 15, cy0), f"{nm}: {kt} target \u00b7 {kf} filtered", fill=C_FG, font=F["tiny"])
    y += 17 * 3 + 6
    n_no3d = int((~zm[idx]).sum()) if len(idx) else 0
    n_fl = int(np.sum(np.abs(bottom[np.isfinite(bottom)]) > FLOAT_M)) if len(idx) else 0
    d.text((xs, y), f"sanity: size outliers {len(flags)} \u00b7 |bottom| > {FLOAT_M} m {n_fl} \u00b7 "
                    f"duplicate pairs (BEV IoU > {DUP_IOU}) {len(dups)} \u00b7 no 3-D label {n_no3d}",
           fill=C_WARN if (flags or dups or n_fl) else C_DIM, font=F["tiny"])
    y += 17
    for f in flags[:3]:
        d.text((xs, y), f"  {f[1]} @ {f[3]:.0f} m: {'; '.join(f[2])}", fill=C_WARN, font=F["tiny"])
        y += 16
    return cv


def draw_title(path, cs, ci, n_clips, F, PAL, AGC):
    from PIL import Image, ImageDraw
    cv = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(cv)
    y = 60
    d.text((60, y), f"GT VALIDATION \u00b7 clip {ci + 1}/{n_clips} \u00b7 sha12 {cs['clip_sha12']} "
                    f"\u00b7 {cs['role']}", fill=C_FG, font=F["title"])
    y += 60
    g, t = cs["gt_per_labelled_window"], cs["targets_per_labelled_window"]
    fl = cs["filter"]
    lines = [
        f"{cs['n_windows']} windows ({cs['n_windows_no_label']} NO_LABEL, {cs['n_windows_labelled_clear']} "
        f"labelled clear) \u00b7 {cs['n_gt_boxes']:,} GT boxes \u00b7 {cs['n_tracks']} tracks",
        (f"GT per labelled window: mean {g.get('mean', 0):.1f}, median {g.get('median', 0):.0f}, max "
         f"{g.get('max', 0):.0f} \u00b7 trainer targets per window: mean {t.get('mean', 0):.1f}, max "
         f"{t.get('max', 0):.0f}") if g.get("n") else "no labelled window",
        f"removed by visible_target_filter: {fl['frac_removed'] * 100:.1f} % \u00b7 by first reason: "
        + " \u00b7 ".join(f"{REASON_SHORT[k]} {v['frac_of_gt'] * 100:.1f} %"
                          for k, v in enumerate(fl["by_primary_reason"].values())),
        f"outside the 120\u00b0 field: {cs['frac_outside_120deg_fov'] * 100:.1f} % of GT",
        f"size outliers vs literal class norms: {cs['size_outliers_vs_literal_norms']['n']} "
        f"({cs['size_outliers_vs_literal_norms']['frac_of_gt'] * 100:.2f} %) \u00b7 |bottom| > {FLOAT_M} m: "
        f"{cs['box_bottom_cz_minus_h_over_2']['n_abs_gt_0p5m']} of "
        f"{cs['box_bottom_cz_minus_h_over_2']['n_with_3d_label']:,} 3-D-labelled",
        f"duplicate GT pairs (BEV IoU > {DUP_IOU}): {cs['duplicates_bev_iou_gt_0p5']['n_pairs']} "
        f"({cs['duplicates_bev_iou_gt_0p5']['n_pairs_both_targets']} with both boxes trainer targets)",
        "GT by class: " + ", ".join(f"{k} {v:,}" for k, v in list(cs["gt_by_class"].items())[:6]),
    ]
    for ln in lines:
        d.text((60, y), ln, fill=C_FG, font=F["med"])
        y += 40
    y += 20
    for ln in ("HOW TO READ A FRAME", "camera A (top) = what the boxes video showed; camera B = the SAME image "
               "with ONLY the GT", "solid box = a trainer target; dashed = GT the trainer's filter removes "
               "(the reason is printed)", "BEV B = GT only, wide field; the green outline is the only region "
               "the model is trained to put boxes in"):
        d.text((60, y), ln, fill=C_DIM if ln != "HOW TO READ A FRAME" else C_WARN, font=F["hud"])
        y += 28
    cv.save(path, compress_level=3)


if __name__ == "__main__":
    sys.exit(main())
