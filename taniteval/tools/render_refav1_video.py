#!/usr/bin/env python3
"""render_refav1_video.py — the four-panel refav1 planner reel.

⛔ **READ `taniteval/tools/REFAV1_ARM.md` §3a FIRST.** It is the definition of
what refav1's `cl` arm *is*, and the single fact this reel exists to make
visible: MEASURED at step 21,109 over 141 clips / 282 windows, `cl` is
**bit-identical to the constant-velocity floor `ha0` on 270/282 = 95.74 %** of
windows, emits **kappa identically zero on 282/282 = 100 %**, and over the whole
eval split produces exactly **two** distinct plans (`a = 0` and `a = -1.5`).
A four-family table cannot show that; a picture can, and must not hide it.

⭐ **THEREFORE `ha0` IS DRAWN ON EVERY FRAME, BESIDE THE MODEL.** If the orange
model path and the white floor path are the same line, the viewer sees it
without trusting a caption. That is the whole point of the reel, and it is why
`ha0` is not optional here.

WHAT THIS TOOL DOES AND DOES NOT DO
===================================
This file draws pixels. It **invents no number and runs no model**: every
trajectory, every decision-head argmax and every planner provenance field is read
out of a dump written by `taniteval/tools/refav1_arm.py` — the same tool, the
same STRICT checkpoint load, the same planner config, that produced the scored
eval record. A renderer that re-planned its own way could show a different model
from the one the eval scores.

It **imports** `render_refcv3_video.py` for the camera model (`CylProjector`),
the polyline/densify/text helpers and the per-clip extrinsics loader, so there is
ONE projection implementation in the programme rather than two that can drift.

THE FIVE THINGS THIS RENDERER REFUSES TO GET WRONG
==================================================
**(a) Frames are bridged LOCALLY (C79).** The overlay is drawn on the frame
decoded from the clip's own `*.v2ep.pt`, at the raw index `2t` that the window at
cache index `t` observes last — the same bytes the fp8 encoder consumed. Nothing
is re-fetched or re-rendered between scoring and drawing, and the upscale is
**NEAREST**, so no pixel is invented.

**(b) The codec is read from the `codec` FIELD, never from the buffer's name.**
The payload's buffer is called `jpeg_buf` and its `codec` says `png` (magic
`0x89 0x50`). Decoding by the name raises — or worse, on a pre-allocated output,
leaves zeros and exits 0. That is the documented E-DETECT-1 trap.

**(c) The projection is the clip's own, and it is CYLINDRICAL.** Read from each
`*.v2ep.pt`'s own `frame` dict (256x640, `f_ref` 305.577), never assumed. The
pinhole formula on this raster implies a 92.6 deg field where the rig is
`camera_front_wide_120fov`. `CylProjector` refuses a projection it cannot
express. Extrinsics are **per-clip and MEASURED** from PhysicalAI's own
`calibration/sensor_extrinsics`; without them the camera overlay is DISABLED and
says so on the frame. ⭐ The frame carries its own falsifier: the horizon row
predicted from the clip's extrinsics is drawn as a thin line — if it does not sit
where the road meets the sky, the projection is wrong and the viewer can see it.

**(d) Between replans the plan is HELD and TRANSFORMED, never re-invented.**
refav1 plans once per scored window (here every `--window-stride` cache steps).
On the frames in between, the last plan is carried into the current ego frame
through the RECORDED poses — which is exactly open-loop execution of an MPC plan.
The frame states the age of the plan it is drawing. ⛔ No plan is interpolated
into existence, and a frame before the clip's first scored window draws none.

**(e) Every arm here is OPEN LOOP** (PI ruling 2026-09-02). The model controls
nothing; the ego data keeps arriving from the recording. `T0`/`T1` are the
doctrine's *conditioning* labels, not loop labels. The phrase "closed loop"
describes nothing in this reel.

ACTION-UNIT NOTE (RETRACTION_LOG #15)
====================================
`actions[:, 0]` in the corpus is a road-wheel **STEER ANGLE** `arctan(L*kappa)`
at L = 2.9 m, ~2.9x the magnitude of kappa. The dump this reel reads was rolled
under `--action-units kappa` (the legacy reading), so the `ha` and `ol` arms
over-rotate by that known factor. `cl` and `ha0` are unaffected — both are
exactly zero in either unit — and they are the two arms this reel compares.

USAGE
=====
    python taniteval/tools/render_refav1_video.py \
        --dump-dir  <refav1_arm.py --dump-dir output> \
        --episodes  <dir of the chosen *.v2ep.pt> \
        --extrinsics extrinsics.json \
        --out       reel.mp4  --fps 10 --expect-step 21109

Verify the result with `taniteval/tools/verify_mp4.py`, which DECODES IT BACK.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name: str, path: str):
    if not os.path.exists(path):
        sys.exit(f"[render_refav1] required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: ⭐ THE CAMERA MODEL AND THE TEXT/GEOMETRY HELPERS, IMPORTED — never
#: re-implemented. One projection in the programme, not two that drift.
rv = _load_by_path("render_refcv3_for_refav1",
                   os.path.join(_HERE, "render_refcv3_video.py"))

import torch                                                       # noqa: E402
import torchvision.io as tvio                                      # noqa: E402
from PIL import Image, ImageDraw                                   # noqa: E402

from tanitad.models.v6 import (tactical_lat_actions,               # noqa: E402
                               tactical_lon_actions_v)
from tanitad.refs.refb import NAV_COMMANDS, ROUTE_CLASSES          # noqa: E402
from tanitad.viz_standard import VizElement, check_frame           # noqa: E402

CylProjector = rv.CylProjector
polyline, densify, font, fit, wrap = rv.polyline, rv.densify, rv.font, rv.fit, rv.wrap
load_extrinsics = rv.load_extrinsics

DT_CACHE = 0.2          # one cache step
DT_FRAME = 0.1          # one raw frame
CAM_UP = 2              # 256x640 -> 512x1280

# --------------------------------------------------------------------------- #
# COLOUR SEMANTICS — one meaning per colour, in EVERY panel                    #
# --------------------------------------------------------------------------- #
C_GT = (110, 231, 138)          # GROUND TRUTH — green, everywhere, always
C_CL = (255, 158, 61)           # refav1's OWN plan (`cl`) — orange
C_HA0 = (236, 240, 246)         # the TRIVIAL FLOOR `ha0` — white
C_HA = (120, 140, 168)          # the hold-action control `ha` — slate
C_GIVEN = (240, 190, 90)        # a GIVEN INPUT (the nav token) — amber
C_BG = (9, 12, 17)
C_PANEL = (16, 21, 29)
C_GRID = (34, 42, 53)
C_FG = (233, 238, 245)
C_DIM = (140, 152, 168)
C_WARN = (245, 180, 90)
C_OK = (110, 231, 138)
C_BAD = (248, 113, 113)

# --------------------------------------------------------------------------- #
# LAYOUT — 1920x1122, the same canvas as the refcv3 reel                       #
# --------------------------------------------------------------------------- #
PAD = 12
W_LEFT, H_CAM = 1280, 512
W_RIGHT = 604
W_TOT = PAD + W_LEFT + PAD + W_RIGHT + PAD          # 1920
Y_BAN, H_BAN = 0, 74
Y_CAM = Y_BAN + H_BAN + 8
Y_TAC, H_TAC = Y_CAM + H_CAM + 12, 254
Y_STR, H_STR = Y_TAC + H_TAC + 12, 152
H_BEV = (Y_STR + H_STR) - Y_CAM
Y_HUD = Y_STR + H_STR + 12
H_HUD = 74
H_TOT = Y_HUD + H_HUD + PAD                          # 1122

PLAN_SOURCE_NAMES = ("cem", "baseline:cv", "baseline:hold_v0",
                     "baseline:proposal", "baseline:decel_1.5")


def _p(*a):
    print(*a, flush=True)


# =========================================================================== #
# THE DUMP                                                                    #
# =========================================================================== #
def read_dump(dump_dir: str) -> dict:
    """Every array this reel draws, read once, with its provenance."""
    man_p = os.path.join(dump_dir, "manifest.json")
    with open(man_p, encoding="utf-8") as fh:
        man = json.load(fh)
    eps = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not eps:
        sys.exit(f"[render_refav1] no ep*.npz under {dump_dir}")
    out = {}
    for p in eps:
        d = np.load(p)
        dp = np.load(os.path.join(dump_dir, "decisions", os.path.basename(p)))
        fi = int(os.path.basename(p)[2:5])
        name = None
        for e in man.get("episodes", []):
            if int(e.get("file_index", -1)) == fi:
                name = e.get("name")
        if name is None:
            sys.exit(f"[render_refav1] manifest has no episode for {p}")
        out[name] = dict(
            ws=d["ws"].astype(int), v0=d["v0"].astype(float),
            g=d["g"].astype(float), cl=d["cl"].astype(float),
            ha0=d["ha0"].astype(float), ha=d["ha"].astype(float),
            ol=d["ol"].astype(float),
            ctrl=dp["cl_controls"].astype(float),
            src=dp["plan_source_cl"].astype(int),
            cost=dp["plan_cost_cl"].astype(float),
            neval=dp["plan_neval_cl"].astype(int),
            nav=dp["nav_cmd"].astype(int), nav_valid=dp["nav_valid"].astype(bool),
            glat=dp["goal_lat_cl"].astype(int), glon=dp["goal_lon_cl"].astype(int),
            lat_lab=dp["lat_label"].astype(int), lon_lab=dp["lon_label"].astype(int),
            route_lab=dp["route_label"].astype(int),
            lat_t=dp["lat_pred_nav_true"].astype(int),
            lat_s=dp["lat_pred_nav_shuffled"].astype(int),
            lat_z=dp["lat_pred_nav_zero"].astype(int),
            lon_t=dp["lon_pred_nav_true"].astype(int),
            lon_s=dp["lon_pred_nav_shuffled"].astype(int),
            lon_z=dp["lon_pred_nav_zero"].astype(int),
            rt_t=dp["route_pred_nav_true"].astype(int),
            rt_s=dp["route_pred_nav_shuffled"].astype(int),
            rt_z=dp["route_pred_nav_zero"].astype(int),
            file=os.path.basename(p),
        )
    return {"manifest": man, "eps": out}


# =========================================================================== #
# FRAMES — bridged locally, decoded by the CODEC FIELD                        #
# =========================================================================== #
def episode_payload(episodes_dir: str, name: str) -> dict:
    p = os.path.join(episodes_dir, f"{name}.v2ep.pt")
    if not os.path.exists(p):
        sys.exit(f"[render_refav1] no v2ep for {name} at {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def frame_decoder(pay: dict):
    """A callable raw_index -> uint8 [3, H, W], decoded by the CODEC FIELD.

    ⛔ The buffer is named ``jpeg_buf`` and the codec says ``png``. Trusting the
    NAME is the E-DETECT-1 trap: `decode_jpeg` raises on PNG bytes, and on a
    pre-allocated output that leaves a full-size array of zeros while the job
    exits 0. Read the field."""
    codec = str(pay.get("codec", "jpeg"))
    if codec not in ("png", "jpeg"):
        sys.exit(f"[render_refav1] unknown codec {codec!r} — refusing to guess")
    dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
    buf = pay["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(pay["jpeg_len"].to(torch.int64), 0)])
    magic = bytes(buf[:2].tolist())
    if codec == "png" and magic != b"\x89P":
        sys.exit(f"[render_refav1] codec says png but magic is {magic!r} — "
                 f"refusing (the field and the bytes disagree)")

    def get(i: int):
        a, b = int(offs[i]), int(offs[i + 1])
        return dec(buf[a:b], mode=tvio.ImageReadMode.RGB)
    return get, codec, magic


# =========================================================================== #
# EGO-FRAME TRANSPORT — hold a plan and carry it, never re-invent it          #
# =========================================================================== #
def se2_carry(path_xy: np.ndarray, pose_from: np.ndarray, pose_to: np.ndarray):
    """A path expressed in the ego frame at ``pose_from``, re-expressed in the
    ego frame at ``pose_to``. Poses are ``(x, y, yaw, v)`` in the clip frame.

    This is the ONLY motion this renderer applies, and it is a rigid transform
    of a path the planner already emitted — never an extrapolation."""
    x0, y0, th0 = float(pose_from[0]), float(pose_from[1]), float(pose_from[2])
    x1, y1, th1 = float(pose_to[0]), float(pose_to[1]), float(pose_to[2])
    c0, s0 = math.cos(th0), math.sin(th0)
    world = np.stack([x0 + c0 * path_xy[:, 0] - s0 * path_xy[:, 1],
                      y0 + s0 * path_xy[:, 0] + c0 * path_xy[:, 1]], axis=1)
    c1, s1 = math.cos(th1), math.sin(th1)
    dx, dy = world[:, 0] - x1, world[:, 1] - y1
    return np.stack([c1 * dx + s1 * dy, -s1 * dx + c1 * dy], axis=1)


# =========================================================================== #
# PANELS                                                                      #
# =========================================================================== #
def draw_bev(size, paths, labels_, colours, widths, dashed, span_m, note, F,
             ego_v):
    """The metric BEV — calibration-independent, so it carries the comparison
    even on a clip whose extrinsics are missing."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im)
    m = 46
    x_fwd_max, y_half = span_m, span_m * (w - 2 * m) / (2.0 * (h - 2 * m))
    y_half = max(y_half, 6.0)

    def P(x, y):
        px = m + (w - 2 * m) * (0.5 - y / (2 * y_half))
        py = (h - m) - (h - 2 * m) * (x / x_fwd_max)
        return (px, py)

    for gx in range(0, int(x_fwd_max) + 1, 5):
        py = (h - m) - (h - 2 * m) * (gx / x_fwd_max)
        d.line([(m, py), (w - m, py)], fill=C_GRID, width=1)
        d.text((m + 3, py - 13), f"{gx} m", fill=C_DIM, font=F["micro"])
    for gy in range(-int(y_half), int(y_half) + 1, 5):
        px = m + (w - 2 * m) * (0.5 - gy / (2 * y_half))
        d.line([(px, m), (px, h - m)], fill=C_GRID, width=1)
    d.text((w // 2 - 40, h - m + 4), "+y LEFT   ->   -y RIGHT",
           fill=C_DIM, font=F["micro"])

    # ⭐ ORDER MATTERS: `ha0` is drawn WIDE and UNDERNEATH `cl`, so when the
    # two coincide the floor shows as a halo around the model's line. A viewer
    # must be able to see "these are the same path" rather than "one is
    # missing" — that distinction is the whole finding.
    for path, lab, col, wd, dash in zip(paths, labels_, colours, widths, dashed):
        if path is None:
            continue
        pts = [P(px, py) for px, py in densify(path, 96)]
        if dash:
            for i in range(0, len(pts) - 1, 6):
                d.line(pts[i:i + 4], fill=col, width=wd)
        else:
            d.line(pts, fill=col, width=wd, joint="curve")
        for px, py in path:
            q = P(px, py)
            r = max(3, wd // 2)
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col)

    e = P(0.0, 0.0)
    d.polygon([(e[0], e[1] - 11), (e[0] - 7, e[1] + 7), (e[0] + 7, e[1] + 7)],
              fill=C_FG)
    d.text((m, m + 2), f"METRIC BEV — ego frame, {span_m:.0f} m forward",
           fill=C_FG, font=F["small"])
    # the verdict rides at the TOP of the panel, where nothing can clip it
    for j, ln in enumerate(wrap(d, note, F["micro"], w - 2 * m)[:2]):
        d.text((m, 6 + 17 * j), ln, fill=C_WARN, font=F["micro"])
    d.text((w - m - 118, m + 4), f"v0 {ego_v:5.2f} m/s", fill=C_DIM,
           font=F["micro"])
    return im


def draw_card(spec, F, foot):
    """A full-screen text card. TEXT ONLY, on purpose: a card states what was
    MEASURED and where, and never draws a figure a viewer could mistake for a
    result computed on the frames around it."""
    im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(im)
    style = {"h1": (F["h1"], C_FG, 44), "h2": (F["h2"], C_FG, 32),
             "body": (F["body"], C_FG, 28), "ok": (F["body"], C_OK, 28),
             "warn": (F["body"], C_WARN, 28), "bad": (F["body"], C_BAD, 28),
             "dim": (F["micro"], C_DIM, 22), "gap": (F["micro"], C_DIM, 18)}
    y = 96
    d.text((90, y), spec.get("title", ""), fill=C_FG, font=F["h1"])
    y += 62
    d.line([(90, y), (W_TOT - 90, y)], fill=C_GRID, width=2)
    y += 22
    for item in spec.get("lines", []):
        text, st = (item if isinstance(item, (list, tuple)) else (item, "body"))
        f, col, step = style.get(st, style["body"])
        if st == "gap":
            y += step
            continue
        for ln in wrap(d, text, f, W_TOT - 200):
            d.text((90, y), ln, fill=col, font=f)
            y += step
    d.text((90, H_TOT - 46), foot, fill=C_DIM, font=F["micro"])
    return im


def head_row(d, x, y, w, title, classes, preds, labels_, cond_names, F):
    """One decision head: its argmax under each conditioning, against the label.

    ⛔ The dump stores argmaxes, not logits — so this panel shows what the head
    DECIDED under `nav_true` / `nav_shuffled` / `nav_zero`, which is the echo
    evidence itself, and never a probability bar it does not have."""
    d.text((x, y), title, fill=C_FG, font=F["small"])
    yy = y + 24
    lab = labels_
    lab_s = classes[lab] if 0 <= lab < len(classes) else "— (outside the v7.2 ±2.0 s band)"
    d.text((x, yy), "v7.2 GT", fill=C_DIM, font=F["micro"])
    d.text((x + 108, yy), lab_s, fill=C_GT if 0 <= lab < len(classes) else C_DIM,
           font=F["micro"])
    yy += 20
    for cn, pv in zip(cond_names, preds):
        s = classes[pv] if 0 <= pv < len(classes) else f"?{pv}"
        ok = (0 <= lab < len(classes)) and (pv == lab)
        col = C_OK if ok else (C_BAD if 0 <= lab < len(classes) else C_DIM)
        d.text((x, yy), cn, fill=C_GIVEN if cn == "nav_true" else C_DIM,
               font=F["micro"])
        d.text((x + 108, yy), s, fill=col, font=F["micro"])
        if 0 <= lab < len(classes):
            d.text((x + 108 + 190, yy), "MATCH" if ok else "MISS",
                   fill=col, font=F["micro"])
        yy += 20
    return yy


# =========================================================================== #
# MAIN                                                                        #
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--episodes", required=True,
                    help="a dir of *.v2ep.pt — the clips to render")
    ap.add_argument("--extrinsics", default=None,
                    help="clip_id -> MEASURED extrinsic (pai_extrinsics_table.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--span-m", type=float, default=45.0)
    ap.add_argument("--title-frames", type=int, default=14)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--clip-order", default=None,
                    help="comma-separated clip ids, in the order to render")
    ap.add_argument("--notes", default=None,
                    help="JSON: clip_id -> the one-line manoeuvre description")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke cap on TOTAL frames (0 = no cap)")
    ap.add_argument("--cards", default=None,
                    help='JSON {"intro": {"title":…, "lines":[[text, style], …]}, '
                         '"outro": {…}} — full-screen cards. Styles: h1 h2 body '
                         'ok warn bad dim. Every line is TEXT ONLY: a card '
                         'states measurements, it never draws one.')
    ap.add_argument("--card-frames", type=int, default=70,
                    help="frames each full-screen card is held (70 = 7 s @10fps)")
    a = ap.parse_args(argv)

    D = read_dump(a.dump_dir)
    man = D["manifest"]
    step = int(man["model"].get("step", -1))
    if a.expect_step is not None and step != a.expect_step:
        sys.exit(f"[render_refav1] dump is step {step}, --expect-step "
                 f"{a.expect_step}. Refusing to label a reel with a step it is "
                 f"not.")
    ckpt = man["model"].get("ckpt", "?")
    load_rep = man["model"].get("state_dict_load", {})
    strict = (not load_rep.get("missing_keys") and
              not load_rep.get("unexpected_keys"))
    plan_cfg = man.get("plan_cfg", {})
    grid = man.get("grid", {})
    units = man.get("action_units", man.get("units", "kappa"))
    units_short = (units.get("recorded", "kappa") if isinstance(units, dict)
                   else str(units))

    names = list(D["eps"].keys())
    if a.clip_order:
        want = [x.strip() for x in a.clip_order.split(",") if x.strip()]
        missing = [x for x in want if x not in names]
        if missing:
            sys.exit(f"[render_refav1] --clip-order names clips absent from the "
                     f"dump: {missing}")
        names = want + [n for n in names if n not in want]
    notes = {}
    if a.notes:
        with open(a.notes, encoding="utf-8") as fh:
            notes = json.load(fh)

    extr = load_extrinsics(a.extrinsics, names)
    _p(f"[calib] per-clip extrinsics for {len(extr)}/{len(names)} clips"
       + ("" if len(extr) == len(names) else "  ⚠️ camera overlay DISABLED on the rest"))

    F = {"h1": font(30, True), "h2": font(21, True), "body": font(18),
         "small": font(16, True), "micro": font(14), "mono": font(15)}

    frames_dir = a.frames_dir or (os.path.splitext(a.out)[0] + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)

    LAT = list(tactical_lat_actions("v7.0"))
    LON = list(tactical_lon_actions_v("v7.0"))
    ROUTE = list(ROUTE_CLASSES)
    NAVN = list(NAV_COMMANDS)

    cards = {}
    if a.cards:
        with open(a.cards, encoding="utf-8") as fh:
            cards = json.load(fh)
    card_foot = (f"refav1 step {step}  ·  {ckpt}  ·  "
                 f"{'STRICT load' if strict else 'NON-STRICT LOAD'}  ·  "
                 f"dump {os.path.abspath(a.dump_dir)}")

    n_out = 0
    index: list[dict] = []
    stats = dict(frames=0, clips=0, ident=0, scored=0, kzero=0,
                 plans=0, cam_on=0, cam_off=0, skipped_stale=0)
    horizon_s = float(grid.get("horizon_k", 10)) * DT_CACHE
    t_start = time.time()

    if cards.get("intro"):
        im = draw_card(cards["intro"], F, card_foot)
        for _ in range(a.card_frames):
            index.append({"frame": n_out, "kind": "card", "card": "intro"})
            im.save(os.path.join(frames_dir, f"f_{n_out:06d}.png"))
            n_out += 1

    for ci, name in enumerate(names):
        ep = D["eps"][name]
        pay = episode_payload(a.episodes, name)
        get_frame, codec, magic = frame_decoder(pay)
        poses = pay["poses"].numpy().astype(np.float64)      # RAW timeline
        fr_dict = dict(pay["frame"])
        cid = str(pay.get("clip_id", name))
        try:
            proj = CylProjector(fr_dict, extr.get(cid))
        except rv.ProjectionRefused as e:
            _p(f"[clip] {cid}: {e} — camera overlay off")
            proj = None
        cam_on = bool(proj is not None and proj.enabled)
        stats["cam_on" if cam_on else "cam_off"] += 1

        ws = ep["ws"]
        order = np.argsort(ws)
        ws = ws[order]
        for k in ("v0", "g", "cl", "ha0", "ha", "ol", "ctrl", "src", "cost",
                  "neval", "nav", "nav_valid", "glat", "glon", "lat_lab",
                  "lon_lab", "route_lab", "lat_t", "lat_s", "lat_z", "lon_t",
                  "lon_s", "lon_z", "rt_t", "rt_s", "rt_z"):
            ep[k] = ep[k][order]
        f_lo, f_hi = int(2 * ws[0]), min(int(2 * ws[-1]) + 1, poses.shape[0] - 1)
        note = notes.get(cid, "")
        _p(f"[clip {ci+1}/{len(names)}] {cid} codec={codec} magic={magic!r} "
           f"frames {f_lo}..{f_hi} scored_windows={len(ws)} cam={'ON' if cam_on else 'OFF'}")

        # ---------------- title card -------------------------------------- #
        for _ in range(a.title_frames):
            im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
            d = ImageDraw.Draw(im)
            d.text((PAD + 40, 300), f"CLIP {ci + 1} of {len(names)}",
                   fill=C_DIM, font=F["h2"])
            d.text((PAD + 40, 342), cid, fill=C_FG, font=F["h1"])
            for j, ln in enumerate(wrap(d, note, F["h2"], W_TOT - 160)):
                d.text((PAD + 40, 400 + 34 * j), ln, fill=C_WARN, font=F["h2"])
            d.text((PAD + 40, 520),
                   f"refav1  ·  step {step}  ·  OPEN LOOP  ·  "
                   f"{len(ws)} scored windows on this clip",
                   fill=C_DIM, font=F["body"])
            index.append({"frame": n_out, "kind": "title", "clip_id": cid,
                          "clip_order": ci})
            im.save(os.path.join(frames_dir, f"f_{n_out:06d}.png"))
            n_out += 1
            if a.max_frames and n_out >= a.max_frames:
                break
        if a.max_frames and n_out >= a.max_frames:
            break

        # ---------------- the clip ---------------------------------------- #
        for f in range(f_lo, f_hi + 1):
            wi = int(np.searchsorted(ws, f / 2.0, side="right") - 1)
            if wi < 0:
                continue
            t_w = int(ws[wi])
            plan_age_s = (f - 2 * t_w) * DT_FRAME
            # ⛔ Past the plan's own 2.0 s horizon there is no action left to
            # draw. Drawing the stub would show a plan shrinking to nothing and
            # read as a model that stops planning. Skip the frame and COUNT it.
            if plan_age_s > horizon_s - 1e-9:
                stats["skipped_stale"] += 1
                continue
            p_from, p_to = poses[2 * t_w], poses[f]
            carry = (lambda P: se2_carry(P, p_from, p_to))
            g_c = carry(ep["g"][wi])
            cl_c = carry(ep["cl"][wi])
            ha0_c = carry(ep["ha0"][wi])
            ha_c = carry(ep["ha"][wi])
            ctrl = ep["ctrl"][wi]
            ident = bool(np.array_equal(ep["cl"][wi], ep["ha0"][wi]))
            kzero = bool(np.all(ctrl[:, 1] == 0.0))
            src = PLAN_SOURCE_NAMES[int(ep["src"][wi])] \
                if 0 <= int(ep["src"][wi]) < len(PLAN_SOURCE_NAMES) \
                else f"idx{int(ep['src'][wi])}"
            ade_cl = float(np.linalg.norm(ep["cl"][wi] - ep["g"][wi], axis=1).mean())
            ade_ha0 = float(np.linalg.norm(ep["ha0"][wi] - ep["g"][wi], axis=1).mean())
            if f == 2 * t_w:
                stats["scored"] += 1
                stats["ident"] += int(ident)
                stats["kzero"] += int(kzero)

            im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
            d = ImageDraw.Draw(im)
            elements = []

            # ---- banner
            d.rectangle([0, Y_BAN, W_TOT, Y_BAN + H_BAN], fill=C_PANEL)
            d.text((PAD + 6, Y_BAN + 8),
                   f"refav1  ·  step {step}  ·  OPEN LOOP  ·  clip {cid[:8]}  "
                   f"·  t = {t_w * DT_CACHE:5.1f} s  ·  frame {f}",
                   fill=C_FG, font=F["h2"])
            d.text((PAD + 6, Y_BAN + 40),
                   fit(d, f"{'STRICT load (0 missing / 0 unexpected)' if strict else '⛔ NON-STRICT LOAD'}"
                          f"  ·  plan {{samples {plan_cfg.get('n_samples')}, iters "
                          f"{plan_cfg.get('n_iters')}, elites {plan_cfg.get('n_elites')}}}"
                          f"  ·  action-units {units_short}  ·  every line is read "
                          f"from the dump — NO model runs in this renderer",
                       F["micro"], W_TOT - 2 * PAD - 12),
                   fill=C_DIM, font=F["micro"])

            # ---- camera
            x0c, y0c = PAD, Y_CAM
            if cam_on:
                img = get_frame(f).permute(1, 2, 0).numpy()
                cam = Image.fromarray(img).resize(
                    (fr_dict["width"] * CAM_UP, fr_dict["height"] * CAM_UP),
                    Image.NEAREST)                 # NEAREST: no pixel invented
                im.paste(cam, (x0c, y0c))
                dc = ImageDraw.Draw(im)

                def proj_to(path):
                    pts = proj(densify(path, 96), up=CAM_UP)
                    return [None if p is None else (p[0] + x0c, p[1] + y0c)
                            for p in pts]
                # the falsifier: the horizon predicted from THIS clip's extrinsics
                hz = proj(np.array([[4000.0, 0.0]]), up=CAM_UP)
                if hz and hz[0] is not None:
                    yy = hz[0][1] + y0c
                    dc.line([(x0c, yy), (x0c + W_LEFT, yy)],
                            fill=(90, 100, 118), width=1)
                    dc.text((x0c + 8, yy - 18),
                            "horizon predicted from this clip's own extrinsics "
                            "— if it is not on the skyline the projection is wrong",
                            fill=(140, 150, 168), font=F["micro"])
                pts_gt, pts_cl = proj_to(g_c), proj_to(cl_c)
                polyline(dc, pts_gt, C_GT, 7)
                polyline(dc, proj_to(ha_c), C_HA, 3)
                polyline(dc, proj_to(ha0_c), C_HA0, 11)   # the FLOOR, wide…
                polyline(dc, pts_cl, C_CL, 4)             # …the model on top
                # ⛔ AN EMPTY CAMERA PANEL MUST SAY WHY. The projector DROPS a
                # point nearer than 0.5 m radially or behind the image plane
                # rather than clamping it, and the camera sits ~2.0-2.1 m
                # FORWARD of the vehicle origin — so a short remaining plan at
                # low speed lands under the hood and nothing is drawn. Silence
                # there reads as "the model predicted nothing".
                n_vis = sum(1 for q in pts_cl if q is not None)
                if n_vis < 2:
                    dc.text((x0c + 12, y0c + H_CAM - 46),
                            fit(dc, f"the remaining "
                                    f"{max(0.0, 2.0 - plan_age_s):.1f} s of the plan "
                                    f"projects BELOW the image — the camera sits "
                                    f"~2 m forward of the vehicle origin and the "
                                    f"projector drops a point rather than clamping "
                                    f"it. The BEV carries this window.",
                                F["micro"], W_LEFT - 24),
                            fill=C_WARN, font=F["micro"])
                elements.append(VizElement(
                    "camera", "present", kind="derived",
                    value=f"GT (green) + cl (orange) + ha0 (white) projected, "
                          f"plan age {plan_age_s:.1f} s, "
                          f"{n_vis}/{len(pts_cl)} plan points inside the raster",
                    source="CylProjector(v2ep['frame'], per-clip "
                           "sensor_extrinsics) applied to ep*.npz g/cl/ha0, "
                           f"drawn on the v2ep PNG at raw index {f} "
                           f"(C79: bridged locally)"))
            else:
                d.rectangle([x0c, y0c, x0c + W_LEFT, y0c + H_CAM], fill=C_PANEL)
                for j, ln in enumerate(wrap(
                        d, "CAMERA OVERLAY DISABLED — no per-clip extrinsics for "
                           "this clip. The BEV panel carries the comparison; a "
                           "projection guessed from a constant camera height is "
                           "wrong by metres exactly where the trajectory is.",
                        F["body"], W_LEFT - 60)):
                    d.text((x0c + 30, y0c + 60 + 28 * j), ln, fill=C_WARN,
                           font=F["body"])
                elements.append(VizElement(
                    "camera", "unavailable",
                    reason="no MEASURED per-clip extrinsic for this clip; "
                           "refusing to approximate with a constant camera "
                           "height (the three constants in this repo are all "
                           "wrong as a constant)"))
            d.rectangle([x0c, y0c, x0c + W_LEFT, y0c + H_CAM],
                        outline=C_GRID, width=1)

            # ---- BEV
            # the span follows the data: at 24 m/s a 2 s path is ~47 m, at
            # 3 m/s it is ~6 m, and a fixed span makes one of those unreadable.
            reach = max(6.0, float(np.max([np.abs(P).max() for P in
                                           (g_c, cl_c, ha0_c, ha_c)])))
            span = float(min(a.span_m, 10.0 * math.ceil((reach * 1.15) / 10.0)))
            bev = draw_bev((W_RIGHT, H_BEV),
                           [g_c, ha_c, ha0_c, cl_c],
                           ["GT", "ha", "ha0", "cl"],
                           [C_GT, C_HA, C_HA0, C_CL],
                           [6, 4, 11, 4],
                           [False, False, True, False],
                           span,
                           ("cl ≡ ha0 — BIT-IDENTICAL on this window: the "
                            "orange plan IS the white floor"
                            if ident else
                            "cl differs from ha0 — the injected −1.5 m/s² "
                            "brake, not a search result"),
                           F, float(ep["v0"][wi]))
            im.paste(bev, (PAD + W_LEFT + PAD, Y_CAM))
            elements.append(VizElement(
                "bev", "present", kind="derived",
                value=f"GT / cl / ha0 / ha, span {span:.0f} m",
                source="ep*.npz arms g/cl/ha0/ha, carried from the ego frame "
                       "at the scored window into the current ego frame by a "
                       "rigid SE(2) transform of the RECORDED poses"))

            # ---- TACTICAL
            d.rectangle([PAD, Y_TAC, PAD + W_LEFT, Y_TAC + H_TAC], fill=C_PANEL)
            d.text((PAD + 14, Y_TAC + 8),
                   "TACTICAL — the FACTORED heads, argmax under three "
                   "conditionings (the dump stores decisions, not logits)",
                   fill=C_FG, font=F["small"])
            head_row(d, PAD + 24, Y_TAC + 36, 560, "LATERAL head", LAT,
                     [int(ep["lat_t"][wi]), int(ep["lat_s"][wi]), int(ep["lat_z"][wi])],
                     int(ep["lat_lab"][wi]),
                     ["nav_true", "nav_shuffled", "nav_zero"], F)
            head_row(d, PAD + 660, Y_TAC + 36, 560, "LONGITUDINAL head", LON,
                     [int(ep["lon_t"][wi]), int(ep["lon_s"][wi]), int(ep["lon_z"][wi])],
                     int(ep["lon_lab"][wi]),
                     ["nav_true", "nav_shuffled", "nav_zero"], F)
            gl, gn = int(ep["glat"][wi]), int(ep["glon"][wi])
            d.text((PAD + 24, Y_TAC + 160),
                   "the tactical decoder's IMAGINED GOAL, i.e. what the planner "
                   "was asked to achieve:", fill=C_DIM, font=F["micro"])
            d.text((PAD + 24, Y_TAC + 182),
                   f"lat  {LAT[gl] if 0 <= gl < len(LAT) else gl}      "
                   f"lon  {LON[gn] if 0 <= gn < len(LON) else gn}",
                   fill=C_CL, font=F["small"])
            d.text((PAD + 24, Y_TAC + 212),
                   fit(d, "⚠️ the decoder ASKS for turns; the planner emits "
                          f"kappa {'= 0 on this window' if kzero else '≠ 0'} — "
                          "the bottleneck is downstream of the decoder, in the cost",
                       F["micro"], W_LEFT - 60),
                   fill=C_WARN, font=F["micro"])
            elements.append(VizElement(
                "tactical", "present", kind="model_output",
                value=(f"lat {LAT[int(ep['lat_t'][wi])]} / "
                       f"lon {LON[int(ep['lon_t'][wi])]} (nav_true); "
                       f"imagined goal lat {LAT[gl] if 0 <= gl < len(LAT) else gl} "
                       f"/ lon {LON[gn] if 0 <= gn < len(LON) else gn}"),
                source="decisions/ep*.npz lat_pred_*/lon_pred_*/goal_*_cl — "
                       "the heads' own argmax, read from the window only",
                conditioned_on=("nav token (v7.2, a GIVEN INPUT)",)))
            if int(ep["lat_lab"][wi]) >= 0:
                elements.append(VizElement(
                    "tactical_gt", "present", kind="gt_label",
                    value=(f"lat {LAT[int(ep['lat_lab'][wi])]} / "
                           f"lon {LON[int(ep['lon_lab'][wi])]}"),
                    source="v7.2 s2 label record, in-band for this window"))
            else:
                elements.append(VizElement(
                    "tactical_gt", "unavailable",
                    reason="outside the v7.2 record's +-2.0 s band — the label "
                           "is -100 and is NEVER clamped to a neutral class"))

            # ---- STRATEGIC
            d.rectangle([PAD, Y_STR, PAD + W_LEFT, Y_STR + H_STR], fill=C_PANEL)
            d.text((PAD + 14, Y_STR + 8),
                   "STRATEGIC — the route head against its label, and the nav "
                   "token that was FED to it", fill=C_FG, font=F["small"])
            head_row(d, PAD + 24, Y_STR + 32, 560, "ROUTE head", ROUTE,
                     [int(ep["rt_t"][wi]), int(ep["rt_s"][wi]), int(ep["rt_z"][wi])],
                     int(ep["route_lab"][wi]),
                     ["nav_true", "nav_shuffled", "nav_zero"], F)
            nv = int(ep["nav"][wi])
            d.text((PAD + 660, Y_STR + 32), "GIVEN INPUT", fill=C_DIM,
                   font=F["small"])
            d.text((PAD + 660, Y_STR + 58),
                   f"nav token FED:  {NAVN[nv] if 0 <= nv < len(NAVN) else nv}"
                   f"{'' if bool(ep['nav_valid'][wi]) else '  (INVALID)'}",
                   fill=C_GIVEN, font=F["small"])
            d.text((PAD + 660, Y_STR + 86),
                   fit(d, "⛔ route accuracy 1.0000 under nav_true and 0.6383 "
                          "(= the majority rate) under nav_zero: the head "
                          "reproduces its own input.", F["micro"], 560),
                   fill=C_BAD, font=F["micro"])
            d.text((PAD + 660, Y_STR + 106),
                   fit(d, "MEASURED over the 141-clip split, n = 141 labelled "
                          "windows — not on this frame.", F["micro"], 560),
                   fill=C_DIM, font=F["micro"])
            elements.append(VizElement(
                "strategic", "present", kind="model_output",
                value=(f"route {ROUTE[int(ep['rt_t'][wi])]} (nav_true) / "
                       f"{ROUTE[int(ep['rt_s'][wi])]} (nav_shuffled) / "
                       f"{ROUTE[int(ep['rt_z'][wi])]} (nav_zero)"),
                source="decisions/ep*.npz route_pred_* — the route head's own "
                       "argmax under each conditioning",
                conditioned_on=("nav token (v7.2, a GIVEN INPUT)",)))
            elements.append(VizElement(
                "strategic_input", "present", kind="given_input",
                value=NAVN[nv] if 0 <= nv < len(NAVN) else str(nv),
                source="decisions/ep*.npz nav_cmd — the v7.2 nav token FED to "
                       "the strategic head, not anything it predicted"))

            # ---- HUD
            d.rectangle([PAD, Y_HUD, W_TOT - PAD, Y_HUD + H_HUD], fill=C_PANEL)
            a0 = float(ctrl[0, 0])
            d.text((PAD + 14, Y_HUD + 6),
                   f"PLAN  a ≡ {a0:+.3f} m/s²   kappa ≡ {float(ctrl[0,1]):+.4f} 1/m   "
                   f"source \"{src}\"   cost {float(ep['cost'][wi]):.3e}   "
                   f"candidates {int(ep['neval'][wi])}   plan age {plan_age_s:4.1f} s",
                   fill=C_FG, font=F["small"])
            v = "cl ≡ ha0  (BIT-IDENTICAL — the model's path IS the trivial floor)" \
                if ident else "cl ≠ ha0  (the −1.5 m/s² injected brake, not a search result)"
            d.text((PAD + 14, Y_HUD + 30), v,
                   fill=C_BAD if ident else C_WARN, font=F["small"])
            d.text((PAD + 14, Y_HUD + 52),
                   f"ADE over this 2.0 s window:  cl {ade_cl:5.3f} m    "
                   f"ha0 {ade_ha0:5.3f} m    ·  GREEN = ground truth, ORANGE = "
                   f"refav1's plan, WHITE DASHED = ha0 (a = 0, kappa = 0 at the "
                   f"measured v0), SLATE = ha hold-action",
                   fill=C_DIM, font=F["micro"])
            elements.append(VizElement(
                "ade", "present", kind="derived",
                value=f"cl {ade_cl:.3f} m / ha0 {ade_ha0:.3f} m",
                source="mean ||arm - g||_2 over the K=10 waypoints of this "
                       "window at dt 0.2 s, from ep*.npz"))

            check_frame(elements, where=f"render_refav1_video:{cid}:{f}")
            im.save(os.path.join(frames_dir, f"f_{n_out:06d}.png"))
            # ⭐ the frame INDEX: which clip / raw frame / scored window every
            # output frame is, so a representative still can be chosen by a
            # MEASURED property instead of by eye — and so a still committed to
            # the repo can be traced back to the window it shows.
            index.append({"frame": n_out, "kind": "clip", "clip_id": cid,
                          "clip_order": ci, "raw_frame": f,
                          "t_cache": t_w, "t_s": round(t_w * DT_CACHE, 2),
                          "plan_age_s": round(plan_age_s, 2),
                          "cl_equals_ha0": ident, "kappa_zero": kzero,
                          "plan_source": src, "a_mps2": float(ctrl[0, 0]),
                          "kappa_1pm": float(ctrl[0, 1]),
                          "ade_cl_m": round(ade_cl, 4),
                          "ade_ha0_m": round(ade_ha0, 4),
                          "v0_mps": round(float(ep["v0"][wi]), 3),
                          "gt_lat_extent_m": round(float(np.abs(g_c[:, 1]).max()), 3),
                          "camera": bool(cam_on)})
            n_out += 1
            stats["frames"] += 1
            if a.max_frames and n_out >= a.max_frames:
                break
        stats["clips"] += 1
        if a.max_frames and n_out >= a.max_frames:
            break

    if cards.get("outro") and not (a.max_frames and n_out >= a.max_frames):
        im = draw_card(cards["outro"], F, card_foot)
        for _ in range(a.card_frames):
            index.append({"frame": n_out, "kind": "card", "card": "outro"})
            im.save(os.path.join(frames_dir, f"f_{n_out:06d}.png"))
            n_out += 1

    idx_path = os.path.splitext(a.out)[0] + "_frame_index.json"
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump({"tool": "render_refav1_video.py", "step": step, "fps": a.fps,
                   "n_frames": n_out, "ckpt": ckpt, "dump_dir":
                   os.path.abspath(a.dump_dir), "stats": stats,
                   "frames": index}, fh, indent=1)
    _p(f"[index ] {idx_path}")
    _p(f"[frames] {n_out} written to {frames_dir} in {time.time()-t_start:.0f} s")
    _p(f"[shape ] scored windows drawn {stats['scored']}  "
       f"cl≡ha0 on {stats['ident']}  kappa≡0 on {stats['kzero']}  "
       f"frames skipped past the 2.0 s plan horizon {stats['skipped_stale']}")

    # ---- encode: full quality, then a small copy FROM THE FRAMES ---------- #
    ff = "ffmpeg"
    full = a.out
    small = os.path.splitext(a.out)[0] + "_small.mp4"
    pat = os.path.join(frames_dir, "f_%06d.png")
    cmd_full = [ff, "-y", "-v", "error", "-framerate", str(a.fps), "-i", pat,
                "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", full]
    # ⚠️ the small copy is RE-ENCODED FROM THE ORIGINAL FRAMES, never
    # transcoded from `full`: a transcode stacks a second generation of loss.
    cmd_small = [ff, "-y", "-v", "error", "-framerate", str(a.fps), "-i", pat,
                 "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow",
                 "-crf", "31", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 small]
    for cmd, tag in ((cmd_full, "full"), (cmd_small, "small")):
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.exit(f"[render_refav1] ffmpeg ({tag}) failed: {r.stderr[:800]}")
        _p(f"[encode] {tag}: {cmd[-1]}  {os.path.getsize(cmd[-1]):,} B")

    if not a.keep_frames:
        keep = {0, n_out // 2, n_out - 1}
        keep |= {int(r["frame"]) for r in index if r.get("kind") == "title"}
        for i, p in enumerate(sorted(glob.glob(os.path.join(frames_dir, "f_*.png")))):
            if i not in keep:
                os.remove(p)
    _p("[done] now verify BOTH files by DECODING THEM BACK: "
       "python taniteval/tools/verify_mp4.py "
       f"{full} {small}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
