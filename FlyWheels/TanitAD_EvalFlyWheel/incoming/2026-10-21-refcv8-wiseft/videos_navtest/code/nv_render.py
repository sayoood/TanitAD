"""nv_render.py -- NavSim navtest video reels: refcv8 WiSE blend theta_0.5 vs refcv8 EMA-48,852 vs model-free prior vs the human future.  CPU only.

One frame per SCENE (token): the camera t0 frame (416x1024 CYLINDRICAL, f_ref 488.924, column linear in azimuth) with the human future (green ribbon) and the three
8-pose plans projected onto the road plane, a metric BEV pane, a per-arm PDMS/NC/DAC/TTC/EP/C table with the best arm highlighted, and a PDMS ribbon along the clip.
Camera geometry = the bridge's own virtual camera (frames416.rig_record: CAM_F0 mount + the stitch's rotation M, road plane z = median from road_plane_navhard_warmup_logs.json,
exactly what the model's BEV lift used) through r8v_geom.CylCam (cylindrical, NOT pinhole).

usage:
  nv_render.py check   --out DIR                      3 projection-check PNGs (camera pane only, human path + plans)
  nv_render.py still   --clip I --scene K --out F.png  one composed frame
  nv_render.py clip    --clip I --out F.mp4            one clip (intro card + scenes)
  nv_render.py cards   --out DIR                       title + closing cards as mp4
  nv_render.py reel    --out F.mp4                     concat title + clips + closing (clips must already exist in --clips-dir)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8v_geom as G  # noqa: E402
import nv_select as S  # noqa: E402

WORK = Path("C:/Users/Admin/refcv8_pf/navsim_video/work")
OUTROOT = Path("C:/Users/Admin/refcv8_pf/navsim_video")
ROAD_Z = -0.3557311968663991          # road_plane_navhard_warmup_logs.json summary.median (the value the bridge's lift used)
W_, H_ = 1920, 1080
TITLE_H = 54
CAM_XY = (0, TITLE_H)
CAM_W, CAM_H = 1280, 520              # 1.25 x of 1024 x 416
CAM_S = 1.25
BEV_XY, BEV_W, BEV_H = (1280, TITLE_H), 640, 520
ROW2_Y = TITLE_H + CAM_H              # 574
PPM = 8.0                             # BEV pixels per metre
SS = 2                                # camera-overlay supersampling

BG = (14, 16, 20)
PANEL = (22, 25, 31)
EDGE = (60, 66, 78)
FG = (232, 235, 240)
DIM = (150, 158, 170)
GT = (0, 214, 90)
THETA = (255, 140, 0)
EMA = (232, 62, 200)
PRIOR = (170, 170, 170)
GOLD = (255, 205, 60)
RED = (235, 70, 70)
FONT_DIR = Path("C:/Windows/Fonts")

HOLD_S = 0.8                           # each scene is held 0.8 s (real spacing 0.5 s => 1.6x slow motion)
FPS = 10
NAV_NAMES = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}
TIER_TXT = ("NavSim open-loop PDM-score benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. "
            "NOT leaderboard-submitted. One training seed (n=1).")
TITLE_EXACT = ("refcv8 on NavSim navtest (12,146 scenes, open loop, one training seed). WiSE blend theta_0.5 (half refcv7, half refcv8 weights, no training): "
               "PDMS 72.78. refcv8 as trained (EMA-48,852, raw plan): 57.84; its shipped re-timed stage: 56.67. refcv7: 70.20. Model-free prior: 58.66.")
TITLE_RT_LINE = "In this set the refcv8 arm on screen is the shipped re-timed stage (56.67)."


def ema_label(D, short=False):
    """the stage-naming label of the refcv8 EMA arm on screen (coordinator 2026-10-21: raw plan = R8_LEGAL 57.84; shipped re-timed = R8_RT 56.67)"""
    if short:
        return "raw plan" if D.ema_arm == "LEGAL" else "shipped, re-timed"
    return "refcv8 EMA-48,852 (raw plan)" if D.ema_arm == "LEGAL" else "refcv8 EMA-48,852 (shipped, re-timed)"



def font(size, bold=False, mono=False):
    names = (["DejaVuSansMono-Bold.ttf", "consolab.ttf"] if bold else ["DejaVuSansMono.ttf", "consola.ttf"]) if mono else \
            (["DejaVuSans-Bold.ttf", "segoeuib.ttf"] if bold else ["DejaVuSans.ttf", "segoeui.ttf"])
    for n in names:
        p = FONT_DIR / n
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


_F = {}


def F(size, bold=False, mono=False):
    k = (size, bold, mono)
    if k not in _F:
        _F[k] = font(size, bold, mono)
    return _F[k]


def ffmpeg_exe():
    p = shutil.which("ffmpeg")
    if p:
        return p
    c = list(Path("C:/Users/Admin/AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    if not c:
        raise SystemExit("ffmpeg not found")
    return str(c[0])


def open_ffmpeg(out, fps=FPS, crf=18, threads=4):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads),
           "-g", str(fps * 2), "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def dashed(d, pts, fill, width, dash=14, gap=9):
    on, left = True, dash
    for a, b in zip(pts[:-1], pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6:
            continue
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                p0 = (a[0] + (b[0] - a[0]) * pos / seg, a[1] + (b[1] - a[1]) * pos / seg)
                p1 = (a[0] + (b[0] - a[0]) * (pos + step) / seg, a[1] + (b[1] - a[1]) * (pos + step) / seg)
                d.line([p0, p1], fill=fill, width=width)
            pos += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def pdms_color(v):
    v = float(min(1.0, max(0.0, v)))
    stops = [(0.0, (205, 45, 45)), (0.5, (232, 190, 50)), (1.0, (45, 195, 95))]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return tuple(int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(3))
    return stops[-1][1]


# ------------------------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self, ema_arm="LEGAL"):
        self.ema_arm = ema_arm
        B = S.load_bars()
        self.sc = {"W": B.read_scores_csv(S.CSV["W"]), "E": B.read_scores_csv(S.CSV_EMA[ema_arm]), "P": B.read_scores_csv(S.CSV["P"])}
        self.plans = {}
        self.plan_src = {}
        for k, p in (("W", S.SEAMS["W"]), ("E", S.SEAMS["E_" + ema_arm]), ("P", S.SEAMS["P"])):
            z = np.load(p, allow_pickle=False)
            toks = [str(t) for t in z["token"]]
            self.plans[k] = (z["poses"], {t: i for i, t in enumerate(toks)})
            self.plan_src[k] = sorted(set(str(s) for s in z["source"]))
        inp = json.load(gzip.open(S.INP, "rt", encoding="utf-8"))["tokens"]
        self.inp = inp
        self.rigs = json.load(open(WORK / "rigs.json"))
        sel = json.load(open(HERE.parent / "selection.json"))[ema_arm]["rows"]
        self.sel = {r["log"]: r for r in sel}
        self.wanted = json.load(open(WORK / "wanted.json"))          # log -> tokens, in LEGAL-rank order; frames_{i} follows this order
        self.logs = list(self.wanted.keys())
        self.n_eligible = len(sel)

    def plan(self, arm, tok):
        P, ix = self.plans[arm]
        return np.asarray(P[ix[tok]], np.float64)           # [8,3] x,y,heading in the ego frame, 0.5 s .. 4.0 s

    def clip(self, li):
        log = self.logs[li]
        toks = self.wanted[log]
        meta = json.load(open(WORK / "frames" / f"frames_{li}.meta.json"))
        assert [m["token"] for m in meta["tokens"]] == toks
        frames = np.load(WORK / "frames" / f"frames_{li}.npy", mmap_mode="r")
        assert frames.shape[0] == len(toks)
        return Clip(self, li, log, toks, frames)


class Clip:
    def __init__(self, D, li, log, toks, frames):
        self.D, self.li, self.log, self.toks, self.frames = D, li, log, toks, frames
        self.n = len(toks)
        self.sel = D.sel[log]
        self.cams = {}
        for t in toks:
            r = D.rigs[log]["records"][t]
            self.cams[t] = G.CylCam(np.asarray(r["R_virtual_cam_to_lidar"], np.float64),
                                    [r["t_cam_lidar"][0], r["t_cam_lidar"][1], r["t_cam_lidar"][2] - ROAD_Z])
        self.t_s = [(D.inp[t]["timestamps_us"][-1] - D.inp[toks[0]]["timestamps_us"][-1]) / 1e6 for t in toks]
        self.pd = {a: np.array([float(D.sc[a].loc[t, "PDMS"]) for t in toks]) for a in ("W", "E", "P")}
        # BEV scale: fixed per clip so the pane does not zoom between scenes -- the largest of {12,10,8,6} px/m that fits the clip's 98th-percentile forward extent
        ext = []
        for t in toks:
            ext.append(max(float(np.max(self.human(t)[:, 0])), *(float(np.max(D.plan(a, t)[:, 0])) for a in ("W", "E", "P"))))
        e98 = float(np.percentile(ext, 98))
        self.ppm = next((p for p in (12.0, 10.0, 8.0, 6.0) if e98 * 1.08 * p <= BEV_H - 78 - 30), 5.0)
        self.grid_m = 10 if self.ppm >= 8 else 20

    def human(self, t):
        return np.asarray(self.D.inp[t]["human_future_poses"], np.float64)


# ------------------------------------------------------------------------------------------------------------------ camera pane
def project_polyline(cam, xy_dense):
    c, r, v = cam.project_ground(xy_dense)
    return c, r, v


def draw_camera(clip, k, check=False):
    t = clip.toks[k]
    cam = clip.cams[t]
    raw = np.asarray(clip.frames[k])
    base = Image.fromarray(raw).resize((CAM_W, CAM_H), Image.LANCZOS).convert("RGBA")
    ov = Image.new("RGBA", (CAM_W * SS, CAM_H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    s = CAM_S * SS

    def sc(c, r):
        return [(float(a) * s + 0.5 * (s - 1), float(b) * s + 0.5 * (s - 1)) for a, b in zip(c, r)]

    # horizon (a calibration sanity line from the per-clip extrinsics)
    hr = cam.horizon_row() * s
    dashed(d, [(0, hr), (CAM_W * SS, hr)], (255, 255, 255, 60), 2, dash=10, gap=16)
    # human ribbon (1.8 m wide) + centre line
    gt = clip.human(t)[:, :2]
    dn = G.densify(gt, step=0.5)
    if len(dn) >= 2:
        nrm = np.gradient(dn, axis=0)
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        nrm = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        cl, rl, vl = cam.project_ground(dn + 0.9 * nrm)
        cr, rr, vr = cam.project_ground(dn - 0.9 * nrm)
        for i in range(len(dn) - 1):
            if vl[i] and vr[i] and vl[i + 1] and vr[i + 1]:
                d.polygon(sc([cl[i], cl[i + 1], cr[i + 1], cr[i]], [rl[i], rl[i + 1], rr[i + 1], rr[i]]), fill=GT + (46,))
    layers = [("P", PRIOR, 4, True), ("E", EMA, 6, False), ("W", THETA, 6, False)]
    if check:
        layers = [("W", THETA, 4, False)]
    draw_order = [("H", GT, 8, False)] + layers
    for name, col, wid, dash in draw_order:
        xy = gt if name == "H" else clip.D.plan(name, t)[:, :2]
        dn = G.densify(xy, step=0.4)
        c, r, v = cam.project_ground(dn)
        for a, b in G.runs(v):
            if b > a:
                pts = sc(c[a:b + 1], r[a:b + 1])
                if dash:
                    dashed(d, pts, col + (235,), wid * SS // 2, dash=18 * SS // 2, gap=12 * SS // 2)
                else:
                    d.line(pts, fill=col + (255,), width=wid * SS // 2, joint="curve")
        # waypoint dots every 0.5 s (4.0 s end ringed)
        cw, rw, vw = cam.project_ground(xy)
        for i in range(len(xy)):
            if vw[i]:
                x, y = cw[i] * s + 0.5 * (s - 1), rw[i] * s + 0.5 * (s - 1)
                rad = (4 if i < len(xy) - 1 else 7) * SS / 2 * (1.5 if name == "H" else 1.0)
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col + (255,), outline=(0, 0, 0, 255), width=1)
                if i == len(xy) - 1:
                    d.ellipse([x - rad - 3 * SS, y - rad - 3 * SS, x + rad + 3 * SS, y + rad + 3 * SS], outline=col + (255,), width=SS)
    ov = ov.resize((CAM_W, CAM_H), Image.LANCZOS)
    out = Image.alpha_composite(base, ov).convert("RGB")
    return out


# ------------------------------------------------------------------------------------------------------------------ BEV pane
def draw_bev(clip, k):
    t = clip.toks[k]
    im = Image.new("RGB", (BEV_W, BEV_H), (18, 20, 25))
    d = ImageDraw.Draw(im, "RGBA")
    ox, oy = BEV_W / 2, BEV_H - 78
    PPM_ = clip.ppm

    def P(x, y):
        return (ox - y * PPM_, oy - x * PPM_)

    # grid every 10 m (lines + labels along the forward axis)
    gs = clip.grid_m
    fwd_max = int((BEV_H - 78) / PPM_) + gs
    lat_max = int((BEV_W / 2) / PPM_) + gs
    for gx in range(0, fwd_max, gs):
        a, b = P(gx, -lat_max), P(gx, lat_max)
        d.line([a, b], fill=(255, 255, 255, 34 if gx else 80), width=1)
        if gx >= gs:
            d.text((6, P(gx, 0)[1] - 7), f"{gx} m", font=F(12), fill=DIM)
    for gy in range(-(lat_max // gs) * gs, lat_max + 1, gs):
        a, b = P(0, gy), P(fwd_max, gy)
        d.line([a, b], fill=(255, 255, 255, 34 if gy else 80), width=1)
        if gy != 0 and 24 < P(0, gy)[0] < BEV_W - 24:
            d.text((P(0, gy)[0], oy + 18), (f"L {gy} m" if gy > 0 else f"R {-gy} m"), font=F(11), fill=DIM, anchor="mm")
    # camera field of view wedge (virtual camera: +-60 deg about its boresight yaw)
    cam = clip.cams[t]
    cx_, cy_ = float(cam.t[0]), float(cam.t[1])
    yaw = math.atan2(cam.R[1, 2], cam.R[0, 2])
    for sgn in (-1, 1):
        ang = yaw + sgn * math.radians(60)
        a, b = P(cx_, cy_), P(cx_ + 70 * math.cos(ang), cy_ + 70 * math.sin(ang))
        d.line([a, b], fill=(120, 170, 255, 70), width=1)
    # ego footprint: rear axle at the origin, 4.85 x 1.95 m (x from -0.9 to 3.95)
    corners = [(-0.9, 0.975), (3.95, 0.975), (3.95, -0.975), (-0.9, -0.975)]
    d.polygon([P(x, y) for x, y in corners], fill=(90, 110, 150, 150), outline=(190, 205, 235, 255))
    p0 = P(cx_, cy_)
    d.ellipse([p0[0] - 3, p0[1] - 3, p0[0] + 3, p0[1] + 3], fill=(120, 170, 255, 255))
    # trajectories
    gt = clip.human(t)[:, :2]
    for name, col, wid, dash in (("H", GT, 6, False), ("P", PRIOR, 3, True), ("E", EMA, 4, False), ("W", THETA, 4, False)):
        xy = gt if name == "H" else clip.D.plan(name, t)[:, :2]
        pts = [P(0.0, 0.0)] + [P(float(x), float(y)) for x, y in xy]
        if dash:
            dashed(d, pts, col + (240,), wid, dash=10, gap=7)
        else:
            d.line(pts, fill=col + (255,), width=wid, joint="curve")
        for i, (x, y) in enumerate(xy):
            px, py = P(float(x), float(y))
            r = 3 if i < len(xy) - 1 else 6
            d.ellipse([px - r, py - r, px + r, py + r], fill=col + (255,), outline=(0, 0, 0, 255))
    d.text((8, 6), "BEV  top-down, ego frame (rear axle), metres", font=F(15, True), fill=FG)
    d.text((8, BEV_H - 22), "dots every 0.5 s, ring = 4.0 s; no agent boxes in the NavSim eval inputs", font=F(12), fill=DIM)
    return im


# ------------------------------------------------------------------------------------------------------------------ panels
def panel_scores(clip, k, d):
    D = clip.D
    t = clip.toks[k]
    x0, y0 = 0, ROW2_Y
    d.rectangle([x0, y0, 960, H_], fill=PANEL)
    d.text((x0 + 24, y0 + 10), "NavSim PDM score of THIS scene (per-token score CSV)", font=F(20, True), fill=FG)
    cols = ["PDMS", "NC", "DAC", "TTC", "EP", "C"]
    cx = [540, 620, 700, 780, 860, 940]
    for c, x in zip(cols, cx):
        d.text((x, y0 + 46), c, font=F(18, True), fill=DIM, anchor="ra")
    rows = [("W", THETA, "WiSE blend theta_0.5", "half refcv7 + half refcv8 weights, no training"),
            ("E", EMA, "refcv8 EMA-48,852", "raw plan (R8_LEGAL), no retime" if D.ema_arm == "LEGAL" else "shipped re-timed stage (R8_RT)"),
            ("P", PRIOR, "model-free prior", "ha0p, no network")]
    vals = {a: D.sc[a].loc[t] for a in ("W", "E", "P")}
    best = max(float(vals[a]["PDMS"]) for a in vals)
    winners = [a for a in vals if abs(float(vals[a]["PDMS"]) - best) < 1e-9] if best > 1e-9 else []
    ry = y0 + 76
    RH = 64
    for a, col, nm, sub in rows:
        v = vals[a]
        if a in winners:
            d.rounded_rectangle([x0 + 10, ry - 4, 958, ry + RH - 6], radius=8, fill=(60, 52, 18), outline=GOLD, width=3)
        d.rectangle([x0 + 24, ry + 6, x0 + 40, ry + 46], fill=col)
        d.text((x0 + 52, ry + 2), nm, font=F(23, True), fill=FG)
        d.text((x0 + 52, ry + 32), sub, font=F(14), fill=DIM)
        for c, x in zip(cols, cx):
            val = float(v[c])
            fcol = pdms_color(val) if c == "PDMS" else ((RED if val < 0.999 and c in ("NC", "DAC") else FG) if c != "EP" else FG)
            d.text((x, ry + 8), f"{val:.2f}", font=F(30 if c == "PDMS" else 26, True, True), fill=fcol, anchor="ra")
        if a in winners:
            tag = "BEST" if len(winners) == 1 else "BEST (tie)"
            d.text((458, ry + 8), tag, font=F(18, True), fill=GOLD, anchor="ra")
        ry += RH
    # human row
    d.rectangle([x0 + 24, ry + 6, x0 + 40, ry + 46], fill=GT)
    d.text((x0 + 52, ry + 2), "human driver", font=F(23, True), fill=FG)
    d.text((x0 + 52, ry + 32), "the logged future path (reference; not scored per token)", font=F(14), fill=DIM)
    ry += RH + 8
    # scene info
    st = D.inp[t]["ego_statuses"][-1]
    vx, vy = st["ego_velocity"]
    sp = math.hypot(vx, vy)
    ax, ay = st["ego_acceleration"]
    nav = NAV_NAMES[int(np.argmax(st["driving_command"]))]
    lines = [("log", clip.log), ("token", f"{t}   (scene {D.inp[t]['scene_token']})"),
             ("nav cmd", f"{nav}      ego speed {sp:5.1f} m/s = {sp * 3.6:5.1f} km/h      accel {math.hypot(ax, ay):4.1f} m/s^2")]
    for i, (a, b) in enumerate(lines):
        d.text((x0 + 24, ry + i * 26), a, font=F(15, True), fill=DIM)
        d.text((x0 + 110, ry + i * 26), b, font=F(16, mono=True), fill=FG)
    return nav, sp


def panel_ribbon(clip, k, d):
    D = clip.D
    x0, y0, w = 960, ROW2_Y, 960
    d.rectangle([x0, y0, W_, H_], fill=(18, 21, 27))
    d.line([(x0, y0), (x0, H_)], fill=EDGE, width=1)
    d.text((x0 + 24, y0 + 10), f"PDMS along this clip  ({clip.n} scenes, 0.5 s apart)", font=F(20, True), fill=FG)
    lx, rx = x0 + 150, x0 + w - 24
    cw = (rx - lx) / clip.n
    ry = y0 + 52
    RH = 50
    for a, col, nm in (("W", THETA, "theta_0.5"), ("E", EMA, "EMA-48,852"), ("P", PRIOR, "prior")):
        d.rectangle([x0 + 24, ry + 10, x0 + 34, ry + 30], fill=col)
        d.text((x0 + 40, ry + 22), nm, font=F(13, True), fill=FG, anchor="lm")
        for i in range(clip.n):
            d.rectangle([lx + i * cw, ry, lx + (i + 1) * cw - 1.5, ry + RH - 6], fill=pdms_color(clip.pd[a][i]))
        ry += RH + 4
    # cursor
    top, bot = y0 + 46, ry - 2
    d.rectangle([lx + k * cw - 2, top, lx + (k + 1) * cw + 0.5, bot], outline=(255, 255, 255), width=3)
    d.polygon([(lx + (k + 0.5) * cw - 7, top - 1), (lx + (k + 0.5) * cw + 7, top - 1), (lx + (k + 0.5) * cw, top + 10)], fill=(255, 255, 255))
    # colour bar
    by = ry + 6
    for i in range(200):
        d.rectangle([lx + i * 2, by, lx + i * 2 + 2, by + 10], fill=pdms_color(i / 199))
    d.text((lx - 8, by + 5), "PDMS 0", font=F(12), fill=DIM, anchor="rm")
    d.text((lx + 410, by + 5), "1", font=F(12), fill=DIM, anchor="lm")
    # means
    my = by + 62
    hdr = [("theta_0.5", THETA, "W"), ("EMA-48,852", EMA, "E"), ("prior", PRIOR, "P")]
    d.text((x0 + 24, by + 30), "mean PDMS x100", font=F(16, True), fill=FG)
    d.text((x0 + 24, my + 2), "whole clip", font=F(15, True), fill=DIM)
    d.text((x0 + 24, my + 36), f"scenes 1-{k + 1}", font=F(15, True), fill=DIM)
    for j, (nm, col, a) in enumerate(hdr):
        xx = x0 + 260 + j * 230
        d.text((xx, by + 30), nm, font=F(15, True), fill=col)
        d.text((xx, my - 4), f"{100 * clip.pd[a].mean():5.1f}", font=F(28, True, True), fill=FG)
        d.text((xx, my + 32), f"{100 * clip.pd[a][:k + 1].mean():5.1f}", font=F(22, mono=True), fill=DIM)
    dl = 100 * (clip.pd["W"].mean() - clip.pd["E"].mean())
    d.text((x0 + 24, my + 78), f"clip mean (theta_0.5 - EMA) = {dl:+.1f} PDMS points   |   tier {clip.sel['tier']}, rank {clip.sel['rank']}/{clip.D.n_eligible}",
           font=F(17, True), fill=GOLD)
    dd = clip.pd["W"] - clip.pd["E"]
    nw, nl = int((dd > 1e-6).sum()), int((dd < -1e-6).sum())
    d.text((x0 + 24, my + 106), f"per scene: theta_0.5 scores higher than EMA on {nw}, lower on {nl}, equal on {clip.n - nw - nl} of {clip.n}",
           font=F(16), fill=FG)
    # footer
    fy = H_ - 92
    d.text((x0 + 24, fy), "Playback: 1.6x slow motion (scenes are 0.5 s apart in real time; each is held 0.8 s).", font=F(13), fill=DIM)
    yy = fy + 20
    for ln in textwrap.wrap("Tier: " + TIER_TXT, 118):
        d.text((x0 + 24, yy), ln, font=F(13), fill=DIM)
        yy += 17


def compose(clip, k, check=False):
    D = clip.D
    img = Image.new("RGB", (W_, H_), BG)
    d = ImageDraw.Draw(img)
    # header
    d.rectangle([0, 0, W_, TITLE_H], fill=(10, 12, 16))
    t = clip.toks[k]
    d.text((18, 14), "refcv8 on NavSim navtest   |   theta_0.5 vs EMA-48,852 vs prior vs human   |   open-loop PDMS",
           font=F(22, True), fill=FG)
    d.text((W_ - 18, 14), f"{clip.sel['tier']} tier  |  scene {k + 1}/{clip.n}  |  t = +{clip.t_s[k]:.1f} s", font=F(22, True), fill=GOLD, anchor="ra")
    # camera
    cam_img = draw_camera(clip, k, check)
    img.paste(cam_img, CAM_XY)
    d = ImageDraw.Draw(img, "RGBA")
    # legend on the camera pane
    lg_x, lg_y = 12, TITLE_H + 10
    d.rounded_rectangle([lg_x, lg_y, lg_x + 410, lg_y + 128], radius=8, fill=(0, 0, 0, 150))
    items = [("human driver (logged future)", GT, False), ("WiSE blend theta_0.5", THETA, False), (ema_label(D), EMA, False),
             ("model-free prior", PRIOR, True)]
    for i, (nm, col, dsh) in enumerate(items):
        yy = lg_y + 20 + i * 28
        if dsh:
            dashed(d, [(lg_x + 12, yy), (lg_x + 56, yy)], col + (255,), 4, dash=8, gap=6)
        else:
            d.line([(lg_x + 12, yy), (lg_x + 56, yy)], fill=col + (255,), width=6)
        d.text((lg_x + 66, yy), nm, font=F(16, True), fill=FG, anchor="lm")
    # status chip (top right of camera)
    st = D.inp[t]["ego_statuses"][-1]
    sp = math.hypot(*st["ego_velocity"])
    nav = NAV_NAMES[int(np.argmax(st["driving_command"]))]
    chip = f"{nav}   {sp * 3.6:4.0f} km/h"
    wch = d.textlength(chip, font=F(24, True)) + 28
    d.rounded_rectangle([CAM_W - wch - 12, TITLE_H + 10, CAM_W - 12, TITLE_H + 52], radius=8, fill=(0, 0, 0, 170))
    d.text((CAM_W - 12 - wch / 2, TITLE_H + 31), chip, font=F(24, True), fill=FG, anchor="mm")
    # projection stamp (bottom of camera)
    stamp = (f"cylindrical projection (column linear in azimuth), f_ref 488.9 px, 120 deg; ground = road plane z {ROAD_Z:+.3f} m; camera at "
             f"({clip.cams[t].t[0]:.2f}, {clip.cams[t].t[1]:+.2f}, {clip.cams[t].t[2]:.2f} m above road) from the log's own extrinsics (rig {D.rigs[clip.log]['records'][t]['rig_key']})")
    d.rectangle([0, TITLE_H + CAM_H - 24, CAM_W, TITLE_H + CAM_H], fill=(0, 0, 0, 150))
    d.text((10, TITLE_H + CAM_H - 12), stamp, font=F(12), fill=(210, 215, 225), anchor="lm")
    if check:
        return img.crop((0, 0, CAM_W, TITLE_H + CAM_H))
    img.paste(draw_bev(clip, k), BEV_XY)
    d = ImageDraw.Draw(img)
    d.line([(1280, TITLE_H), (1280, ROW2_Y)], fill=EDGE, width=1)
    panel_scores(clip, k, d)
    panel_ribbon(clip, k, d)
    # thin progress bar (clip progress)
    d.rectangle([0, H_ - 5, W_, H_], fill=(30, 34, 42))
    d.rectangle([0, H_ - 5, int(W_ * (k + 1) / clip.n), H_], fill=GOLD)
    return img


# ------------------------------------------------------------------------------------------------------------------ cards
def card_base():
    img = Image.new("RGB", (W_, H_), (8, 10, 14))
    return img, ImageDraw.Draw(img)


def title_card(D):
    img, d = card_base()
    d.text((W_ // 2, 120), "refcv8 on NavSim navtest", font=F(64, True), fill=FG, anchor="mm")
    d.text((W_ // 2, 190), "what the plans look like on the road: WiSE blend vs refcv8 as trained vs prior vs human", font=F(26), fill=DIM, anchor="mm")
    # the exact title text, verbatim, wrapped
    y = 262
    for ln in textwrap.wrap(TITLE_EXACT, 92):
        d.text((W_ // 2, y), ln, font=F(31), fill=FG, anchor="mm")
        y += 44
    if D.ema_arm == "RT":
        d.text((W_ // 2, y + 12), TITLE_RT_LINE, font=F(30, True), fill=GOLD, anchor="mm")
    # number tiles (numbers taken verbatim from the exact text above)
    tiles = [("WiSE theta_0.5", "72.78", THETA, "no training"), ("EMA-48,852 raw plan", "57.84", EMA, "R8_LEGAL"), ("EMA-48,852 shipped", "56.67", (200, 120, 230), "re-timed, R8_RT"),
             ("refcv7", "70.20", (120, 170, 255), ""), ("model-free prior", "58.66", PRIOR, "")]
    tw, gap = 336, 24
    x = (W_ - (5 * tw + 4 * gap)) // 2
    ty = 500
    for nm, v, col, sub in tiles:
        d.rounded_rectangle([x, ty, x + tw, ty + 170], radius=14, fill=(20, 23, 30), outline=col, width=4)
        d.text((x + tw // 2, ty + 32), nm, font=F(23, True), fill=col, anchor="mm")
        d.text((x + tw // 2, ty + 94), v, font=F(62, True, True), fill=FG, anchor="mm")
        d.text((x + tw // 2, ty + 148), "PDMS x100" + (f"   ({sub})" if sub else ""), font=F(15), fill=DIM, anchor="mm")
        x += tw + gap
    d.text((W_ // 2, 730), "Orange = theta_0.5     Magenta = refcv8 EMA-48,852     Grey dashed = model-free prior     Green = human driver", font=F(26, True), fill=FG, anchor="mm")
    if D.ema_arm == "LEGAL":
        arm_txt = ("Per-frame arms: refcv8 = EMA-48,852 raw plan (R8_LEGAL, CSV mean 57.84), like-for-like with theta_0.5, which is also R8_LEGAL-stage; "
                   "the shipped re-timed stage (56.67) is in the alternate set.")
    else:
        arm_txt = "Per-frame arms: refcv8 = EMA-48,852 shipped re-timed stage (R8_RT, CSV mean 56.67); the raw-plan set (57.84) is the primary reel."
    foot = (arm_txt + "  theta_0.5 = WiSE seam (CSV mean 72.78); prior = ha0p (CSV mean 58.66). "
            "Six clips chosen by a stated rule, 40-58 scenes each: they are NOT a representative sample.")
    y = 790
    for ln in textwrap.wrap(foot, 150):
        d.text((W_ // 2, y), ln, font=F(18), fill=DIM, anchor="mm")
        y += 26
    for ln in textwrap.wrap("Tier: " + TIER_TXT, 160):
        d.text((W_ // 2, y + 10), ln, font=F(18), fill=DIM, anchor="mm")
        y += 26
    return img


def clip_card(clip):
    img, d = card_base()
    s = clip.sel
    d.text((W_ // 2, 150), f"{s['tier']} tier   |   rank {s['rank']} of {clip.D.n_eligible} by clip-mean PDMS gain of theta_0.5 over EMA-48,852 ({ema_label(clip.D, True)})", font=F(30, True), fill=GOLD, anchor="mm")
    d.text((W_ // 2, 230), clip.log, font=F(44, True), fill=FG, anchor="mm")
    d.text((W_ // 2, 290), f"{clip.n} consecutive scenes, {clip.t_s[-1]:.1f} s of driving; played at 1.6x slow motion", font=F(26), fill=DIM, anchor="mm")
    x = (W_ - (3 * 420 + 2 * 30)) // 2
    for a, col, nm in (("W", THETA, "WiSE theta_0.5"), ("E", EMA, f"EMA-48,852 ({ema_label(clip.D, True)})"), ("P", PRIOR, "model-free prior")):
        d.rounded_rectangle([x, 390, x + 420, 580], radius=14, fill=(20, 23, 30), outline=col, width=4)
        d.text((x + 210, 428), nm, font=F(26, True), fill=col, anchor="mm")
        d.text((x + 210, 505), f"{100 * clip.pd[a].mean():.1f}", font=F(76, True, True), fill=FG, anchor="mm")
        d.text((x + 210, 556), "mean PDMS x100 over this clip", font=F(16), fill=DIM, anchor="mm")
        x += 450
    dl = 100 * (clip.pd["W"].mean() - clip.pd["E"].mean())
    d.text((W_ // 2, 650), f"theta_0.5 minus EMA-48,852 ({ema_label(clip.D, True)}) on this clip: {dl:+.2f} PDMS points", font=F(32, True), fill=GOLD, anchor="mm")
    d.text((W_ // 2, 710), f"whole log ({s['whole_log_n']} scenes): {s['whole_log_mean_delta_x100']:+.2f} points", font=F(22), fill=DIM, anchor="mm")
    d.text((W_ // 2, 820), "Orange = theta_0.5     Magenta = EMA-48,852     Grey dashed = prior     Green = human driver", font=F(24, True), fill=FG, anchor="mm")
    d.text((W_ // 2, 880), "Left top: camera view with plans projected on the road   |   right top: metric BEV   |   bottom: per-scene scores and the PDMS ribbon", font=F(20), fill=DIM, anchor="mm")
    return img


def closing_card(D, clips):
    img, d = card_base()
    d.text((W_ // 2, 80), "Summary of the six clips", font=F(52, True), fill=FG, anchor="mm")
    x0 = 70
    cols = [("tier", 70), ("log", 170), ("scenes", 760), ("theta_0.5", 900), ("EMA", 1060), ("prior", 1200), ("d(theta-EMA)", 1360), ("whole-log d", 1600)]
    y = 150
    for nm, x in cols:
        d.text((x, y), nm, font=F(20, True), fill=DIM)
    y += 40
    tot = {a: [] for a in "WEP"}
    for c in clips:
        s = c.sel
        vals = {a: 100 * c.pd[a].mean() for a in "WEP"}
        for a in "WEP":
            tot[a].append(c.pd[a])
        row = [s["tier"], c.log, str(c.n), f"{vals['W']:.1f}", f"{vals['E']:.1f}", f"{vals['P']:.1f}", f"{vals['W'] - vals['E']:+.1f}", f"{s['whole_log_mean_delta_x100']:+.1f} (n={s['whole_log_n']})"]
        for (nm, x), v in zip(cols, row):
            d.text((x, y), v, font=F(20, mono=(nm not in ("tier", "log"))), fill=FG if nm != "d(theta-EMA)" else (GOLD if float(v) >= 0 else RED))
        y += 36
    pooled = {a: 100 * np.concatenate(tot[a]).mean() for a in "WEP"}
    n_all = sum(c.n for c in clips)
    d.line([(60, y + 4), (W_ - 60, y + 4)], fill=EDGE, width=2)
    y += 16
    d.text((70, y), "pooled", font=F(20, True), fill=FG)
    d.text((760, y), str(n_all), font=F(20, True, True), fill=FG)
    for (nm, x), v in zip(cols[3:6], [pooled["W"], pooled["E"], pooled["P"]]):
        d.text((x, y), f"{v:.1f}", font=F(20, True, True), fill=FG)
    d.text((1360, y), f"{pooled['W'] - pooled['E']:+.1f}", font=F(20, True, True), fill=GOLD)
    y += 70
    notes = [
        "Read this honestly:",
        f"- These {n_all} scenes are {100 * n_all / 12146:.1f}% of the 12,146 navtest scenes. Only {D.n_eligible} of 136 logs have a >= 40-scene contiguous run, so the clips are the whole eligible set, not a random draw.",
        "- Full navtest (as quoted): WiSE theta_0.5 72.78, EMA-48,852 raw plan 57.84 (R8_LEGAL), its shipped re-timed stage 56.67 (R8_RT), refcv7 70.20, model-free prior 58.66 (PDMS x100, one training seed). Within these six clips the gain over EMA is concentrated in the HIGH and MID tiers; the LOW tier is about even.",
        "- NavSim PDMS is an open-loop, simulated-rollout score: the plans are NOT driving the camera. Frames are the recorded drive; the lines are what each model proposed from it.",
        "- No agent boxes are drawn: the NavSim eval inputs carry no agent tracks. Every per-scene number comes from the per-token score CSVs.",
    ]
    for ln in notes:
        for j, w in enumerate(textwrap.wrap(ln, 150)):
            d.text((70, y), w, font=F(22, bold=(ln == notes[0])), fill=FG if ln == notes[0] else DIM)
            y += 30
    d.text((W_ // 2, H_ - 40), "Tier: " + TIER_TXT[:150] + "...", font=F(14), fill=DIM, anchor="mm")
    return img


# ------------------------------------------------------------------------------------------------------------------ writers
def write_still(p, img, seconds, fps=FPS):
    raw = np.asarray(img, np.uint8).tobytes()
    for _ in range(int(round(seconds * fps))):
        p.stdin.write(raw)


def render_clip_mp4(D, li, out, crf=18, intro_s=3.5, hold_s=HOLD_S, limit=None):
    t0 = time.time()
    clip = D.clip(li)
    p = open_ffmpeg(out, FPS, crf, threads=3)
    write_still(p, clip_card(clip), intro_s)
    n_hold = int(round(hold_s * FPS))
    K = clip.n if limit is None else min(limit, clip.n)
    for k in range(K):
        img = compose(clip, k)
        raw = np.asarray(img, np.uint8).tobytes()
        for _ in range(n_hold):
            p.stdin.write(raw)
        if k % 10 == 0:
            print(f"  clip {li} scene {k + 1}/{clip.n}  {time.time() - t0:.0f}s", flush=True)
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exit {rc}")
    return {"clip": li, "log": clip.log, "tokens": clip.toks[:K], "n": K, "seconds": intro_s + K * n_hold / FPS, "render_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "still", "clip", "cards", "reel"])
    ap.add_argument("--ema-arm", default="LEGAL", choices=["LEGAL", "RT"])
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()
    D = Data(a.ema_arm)
    if a.mode == "still":
        compose(D.clip(a.clip), a.scene).save(a.out)
        print("wrote", a.out)
    elif a.mode == "check":
        os.makedirs(a.out, exist_ok=True)
        picks = [(0, 10), (3, 20), (5, 20)]
        for li, k in picks:
            c = D.clip(li)
            f = Path(a.out) / f"projcheck_clip{li}_scene{k}.png"
            compose(c, k, check=True).save(f)
            print("wrote", f, c.log, c.toks[k])
    elif a.mode == "clip":
        print(json.dumps(render_clip_mp4(D, a.clip, a.out, a.crf, limit=a.limit)))
    elif a.mode == "cards":
        os.makedirs(a.out, exist_ok=True)
        clips = [D.clip(i) for i in range(len(D.logs))]
        for nm, img, sec in (("title", title_card(D), 8.0), ("closing", closing_card(D, clips), 12.0)):
            p = open_ffmpeg(Path(a.out) / f"{nm}.mp4", FPS, a.crf, threads=2)
            write_still(p, img, sec)
            p.stdin.close()
            assert p.wait() == 0
            img.save(Path(a.out) / f"{nm}.png")
            print("wrote", nm)


if __name__ == "__main__":
    main()
