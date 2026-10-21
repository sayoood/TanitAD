"""render_r8_video.py -- refcv8 judgement videos: front camera + metric BEV + map + tactical + metrics, one 1920x1080 composite per frame.  CPU only.

Built on the standard of taniteval/corpus_overlay.py (camera projection + metric BEV inset + text HUD, GT green / model orange in BOTH panes, per-frame
decoded tactical + strategic + ADE) -- EXTENDED to 5 panels and re-pointed at the refcv8 eval139 banked dumps.  The pinhole FlatProjector of
corpus_overlay is NOT used: the eval frames are CYLINDRICAL (416x1024, f_ref 488.924, column linear in azimuth), so the projection is the cylindrical
rig chain of r8v_geom.py (numpy twin of tanitad/data/rig_projection.py), with the per-clip extrinsics.

NATIVE RATE: 10 Hz (poses advance v*0.1 s per row).  The video is written at 10 fps = real time, every RAW frame (201) of the clip.
PLAN REFRESH: the battery's windows sit every 5th row (ws 7..177, stride 5 = 2 Hz), so a plan is a SNAPSHOT.  Between snapshots the last plan is HELD and
RE-EXPRESSED in the current ego frame with the logged poses (its age is printed); outside the window range (first 9 / last ~17 raw frames) no plan exists
and the panels say so -- nothing is interpolated or invented.

LAYERS (each labelled on screen): GT (logged ego path, green) | refcv8 plan (selected, orange) | SHIPPED re-timed plan (RETIME_DEPLOY8, 30k EMA layer, cyan)
| refcv7-50,400 (banked baseline arm of the paired reads, violet dashed) | causal prior (ha0_ext_pose, model-free, grey) | GT agent footprints (B1 3-D join) |
GT map (SAM3 v3 cart_frac).  refcv8's OWN map-head / box-head outputs are NOT banked -> panels say "pending GPU dump (dump_heads_r8.py)".

NAV FEED (2026-10-20, R57): the tactical panel's NAV FED line is the token the roll WROTE as fed (decisions sidecar via r8v_feed), with the per-clip token
beside as "clip nav (not fed)" on a per-frame (*_pf) tree.  A heads dump whose feed / fed tokens disagree with the tree is REFUSED.  On a *_pf tree the
RT (shipped re-timed) layer is the REGISTERED per-frame table or OFF with an on-screen note -- never the clip-fed b208c9c1... table.

    python render_r8_video.py --episode 114 --out E:/refcv8_eval_kit/videos/ep114.mp4 [--tree step40000ema_unc] [--still 120 --png x.png] [--max-frames N]
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
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_data as D  # noqa: E402
import r8v_feed as FEED  # noqa: E402
import r8v_geom as G  # noqa: E402

# --------------------------------------------------------------------------------------------- layout (1920x1080)
W, H = 1920, 1080
TITLE_H = 52
CAM_RECT = (0, TITLE_H, 1280, 520)          # x, y, w, h   (1.25x of 1024x416)
TAC_RECT = (1280, TITLE_H, 640, 520)
ROW2_Y = TITLE_H + 520
ROW2_H = H - ROW2_Y                          # 508
BEV_RECT = (0, ROW2_Y, 340, ROW2_H)
MAP_RECT = (340, ROW2_Y, 340, ROW2_H)
MET_RECT = (680, ROW2_Y, 1240, ROW2_H)
CAM_SCALE = 1.25

# --------------------------------------------------------------------------------------------- palette
BG = (14, 16, 20)
PANEL = (22, 25, 31)
EDGE = (60, 66, 78)
FG = (232, 235, 240)
DIM = (140, 148, 160)
GT = (0, 214, 90)
R8 = (255, 140, 0)
SHIP = (0, 200, 255)
R7 = (190, 130, 255)
PRIOR = (170, 170, 170)
AGENT = (90, 150, 255)
CEIL = (235, 70, 70)
BAD = (235, 70, 70)
GOOD = (0, 214, 90)

MAP_COL = np.array([(40, 42, 48), (60, 90, 150), (255, 255, 255), (255, 230, 80), (255, 150, 200),
                    (230, 70, 70), (200, 120, 40), (120, 170, 120), (10, 10, 10)], np.float64)
MAP_NAMES = ("none", "drivable", "lane", "xwalk", "arrow", "edge", "hatch", "walk", "unseen")

FONT_DIR = Path("C:/Windows/Fonts")


def font(size, bold=False, mono=False):
    names = (["DejaVuSansMono-Bold.ttf", "consolab.ttf"] if bold else ["DejaVuSansMono.ttf", "consola.ttf"]) if mono else \
            (["DejaVuSans-Bold.ttf", "segoeuib.ttf"] if bold else ["DejaVuSans.ttf", "segoeui.ttf"])
    for n in names:
        p = FONT_DIR / n
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TITLE, F_H, F_N, F_S, F_XS = font(19, True), font(16, True), font(15), font(13), font(11)
F_NM, F_SM = font(14, mono=True), font(12, mono=True)


def ffmpeg_exe():
    p = shutil.which("ffmpeg")
    if p:
        return p
    cands = list(Path("C:/Users/Admin/AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    if not cands:
        raise SystemExit("ffmpeg not found")
    return str(cands[0])


def txt(d, xy, s, f=F_N, fill=FG, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def dashed(d, pts, fill, width, dash=14, gap=9):
    """dashed polyline through float pixel points"""
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


# --------------------------------------------------------------------------------------------- the episode
class Episode:
    def __init__(self, bank: D.EvalBank, ci: int, tree_label: str, heads_dir=None, box_p_thr=0.20):
        self.B, self.ci = bank, int(ci)
        self.e = bank.episode(ci)
        self.frames, self.raw_poses = bank.frames(ci)
        e = self.e
        assert np.abs(self.raw_poses[D.K:] - e["poses"]).max() < 1e-4, "raw poses[2:] != decisions ep_poses (row/raw index convention broken)"
        self.cam = G.CylCam.from_extrinsics(bank.extrinsics(ci))
        self.R = len(self.frames)                       # raw frames (201)
        mg = bank.map_gt(ci)
        self.map_frac = None if mg is None else mg[0]
        self.n = len(e["ws"])
        self.ws = e["ws"]
        self.tree_label = tree_label
        self.box_thr = float(box_p_thr)
        self.heads, self.hmap = None, {}
        if heads_dir:
            hp = Path(heads_dir) / f"heads_ep{self.ci:03d}.npz"
            if hp.exists():
                z = np.load(hp)
                self.heads = {k: z[k] for k in z.files}
                self.hmap = {int(w): j for j, w in enumerate(self.heads["ws"])}
                self._check_heads_feed(hp)
        self._precompute()

    def _check_heads_feed(self, hp):
        """The heads dump must have been rolled on the SAME feed as the tree, window by window -- else its map / box / fan panels show another model input."""
        H, e = self.heads, self.e
        if "nav_feed" not in H:
            if e["nav_feed"] == "clip":
                return                                            # a pre-2026-10-20 clip-fed heads dump on a clip-fed tree: consistent
            raise SystemExit(f"{hp.name}: a heads dump without a recorded nav feed (pre-2026-10-20, clip-fed) cannot be shown on the per-frame tree "
                             f"{self.B.tree}; re-dump with dump_heads_r8.py --feed perframe_full")
        if str(H["nav_feed"]) != e["nav_feed"]:
            raise SystemExit(f"{hp.name}: heads nav_feed {str(H['nav_feed'])!r} != tree {self.B.tree} nav_feed {e['nav_feed']!r}")
        wsi = {int(w): i for i, w in enumerate(e["ws"])}
        bad = [int(w) for j, w in enumerate(H["ws"]) if int(w) in wsi and (int(H["nav_cmd_fed"][j]) != int(e["nav_fed"][wsi[int(w)]]) or
               float(np.abs(np.asarray(H["r8_nav"][j], np.float64) - e["r8_nav_fed"][wsi[int(w)]]).max()) > 1e-6)]
        if bad:
            raise SystemExit(f"{hp.name}: fed nav / r8_nav differ from the tree's sidecar at ws {bad[:5]} (n={len(bad)})")

    # ---- per-window GT slots + ADE rows + false-brake flags
    def _precompute(self):
        e, n = self.e, self.n
        ticks = [round(s / D.DT) - 1 for s in D.SLOT_S8]             # 0.5 s -> future tick index 4 ... 6.0 s -> 59
        gt8 = np.zeros((n, 8, 2))
        gv = np.zeros((n,), bool)
        for i in range(n):
            gw = e["gt_future_world"][i][ticks][:, :2]
            gt8[i] = D.to_ego(gw, e["pose_last"][i])
            gv[i] = bool(e["gt_future_valid"][i][ticks].all())
        self.gt8, self.gv6 = gt8, gv
        # controls: the recomputed GT / plan equal the banked panel arrays on the S2 grid
        self.ctl = {"gt8_vs_panel_g4_max_abs_m": float(np.abs(gt8[:, :4] - e["g4"]).max()),
                    "plan8_vs_panel_os4_max_abs_m": float(np.abs(e["plan8"][:, :4] - e["os4"]).max())}
        assert self.ctl["gt8_vs_panel_g4_max_abs_m"] < 2e-3 and self.ctl["plan8_vs_panel_os4_max_abs_m"] < 1e-3, self.ctl

        def ade(P, G8, sl):
            return np.linalg.norm(np.asarray(P)[..., sl, :] - G8[..., sl, :], axis=-1).mean(-1)
        s2, s6 = slice(0, 4), list(D.S6_SLOTS)
        A = {}
        for name, P8 in (("r8", e["plan8"]), ("ship", e.get("ship8")), ("prior", e.get("prior8"))):
            if P8 is None:
                continue
            A[name] = {"a2": ade(P8, gt8, s2), "a6": np.where(gv, ade(P8, gt8, s6), np.nan)}
        # refcv7: the banked baseline arm: S2 4 slots on every window, S6 6 slots on S6 windows only
        r7_2 = np.linalg.norm(e["r7_4"] - e["g4"], axis=-1).mean(-1)
        r7_6 = np.full(n, np.nan)
        r7_8 = [None] * n
        for k, w in enumerate(e["s6_ws"]):
            i = int(np.where(self.ws == w)[0][0])
            r7_6[i] = np.linalg.norm(e["s6_r7"][k] - e["s6_g"][k], axis=-1).mean()
        A["r7"] = {"a2": r7_2, "a6": r7_6}
        self.ade = A
        # false-brake flag per window (battery definition, S6 windows only)
        self.fb = []
        s6map = {int(w): k for k, w in enumerate(e["s6_ws"])}
        for i in range(n):
            w = int(self.ws[i])
            if w in s6map:
                k = s6map[w]
                self.fb.append(D.false_brake(e["plan8"][i], e["s6_v0"][k], e["s6_gt_v6"][k], bool(e["s6_gt_valid"][k])))
            else:
                self.fb.append((None, None) + D.plan_lon_class_one(e["plan8"][i][list(D.S6_SLOTS)], e["v0"][i]))
        # running means (windows up to i)
        self.run = {k: {m: np.array([np.nanmean(v[m][:i + 1]) if np.isfinite(v[m][:i + 1]).any() else np.nan for i in range(n)]) for m in ("a2", "a6")}
                    for k, v in A.items()}

    def hidx(self, i):
        """index into the heads dump of held window i (None when no dump / this window was not dumped)"""
        return None if (self.heads is None or i is None) else self.hmap.get(int(self.ws[i]))

    def head_boxes(self, hj, i, raw):
        """refcv8 BOX head slots with presence >= box_thr, re-framed to the ego frame at `raw`: [(corners [5,2], p)]"""
        H = self.heads
        if H is None or "perc_box_slots_box3d" not in H:
            return []
        p = 1.0 / (1.0 + np.exp(-H["perc_box_slots_presence_logit"][hj].astype(np.float64)))
        sel = np.where(p >= self.box_thr)[0]
        if not len(sel):
            return []
        b = H["perc_box_slots_box3d"][hj][sel].astype(np.float64)               # x y z l w h yaw  (ego frame at the window origin)
        pf, pt = self.raw_poses[int(self.ws[i]) + D.K], self.raw_poses[raw]
        c = D.reframe(b[:, :2], pf, pt)
        yaw = b[:, 6] + (pf[2] - pt[2])
        out = []
        for (cx, cy), yw, l, w, pp in zip(c, yaw, b[:, 3], b[:, 4], p[sel]):
            cs, sn = math.cos(yw), math.sin(yw)
            co = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]])
            out.append((np.stack([cx + cs * co[:, 0] - sn * co[:, 1], cy + sn * co[:, 0] + cs * co[:, 1]], 1), float(pp)))
        return out

    def held(self, raw):
        """index of the window whose snapshot is shown at RAW frame `raw` (None outside the window range); and the plan age in frames."""
        row = raw - D.K
        if row < self.ws[0] or row > self.ws[-1] + 4:
            return None, None
        i = int((row - self.ws[0]) // 5)
        i = min(max(i, 0), self.n - 1)
        return i, int(row - self.ws[i])

    # ---- projection helpers
    def cam_xy(self, xy_ego, step=0.5):
        """ego-frame ground polyline (origin included) -> list of (col,row) polylines in NATIVE frame px"""
        dn = G.densify(xy_ego, step=step)
        c, r, v = self.cam.project_ground(dn)
        return [(np.stack([c[a:b + 1], r[a:b + 1]], 1)) for a, b in G.runs(v) if b > a]

    def held_plan(self, i, raw, P8):
        """a plan snapshot of window i, re-expressed in the ego frame at RAW frame `raw`"""
        w_raw = int(self.ws[i]) + D.K
        return D.reframe(np.asarray(P8, np.float64), self.raw_poses[w_raw], self.raw_poses[raw])

    def gt_future(self, raw, horizon=60):
        hi = min(raw + horizon, self.R - 1)
        if hi <= raw:
            return np.zeros((0, 2))
        return D.to_ego(self.raw_poses[raw + 1:hi + 1, :2], self.raw_poses[raw])

    # ---- panels ---------------------------------------------------------------------------------------------------
    def draw_camera(self, img, raw, i, age):
        x0, y0, w, h = CAM_RECT
        im = Image.fromarray(self.frames[raw]).resize((w, h), Image.BILINEAR)
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        s = CAM_SCALE

        def sc(p):
            return [(float(a) * s + 0.5 * (s - 1), float(b) * s + 0.5 * (s - 1)) for a, b in p]

        # horizon (from the per-clip extrinsics) -- a calibration sanity line
        hr = self.cam.horizon_row() * s
        dashed(d, [(0, hr), (w, hr)], (255, 255, 255, 70), 1, dash=6, gap=10)
        # GT agent footprints (z = 0 plane), non-occluded only
        ag = self.B.agents_at(self.ci, raw)
        if ag is not None:
            arr, _cls = ag
            for a in arr:
                cx, cy, yaw, l, wd, occ = a[:6]
                if occ > 0.5 or not (3.0 < cx < 75.0):
                    continue
                cs, sn = math.cos(yaw), math.sin(yaw)
                corners = np.array([[l / 2, wd / 2], [l / 2, -wd / 2], [-l / 2, -wd / 2], [-l / 2, wd / 2], [l / 2, wd / 2]])
                pts = np.stack([cx + cs * corners[:, 0] - sn * corners[:, 1], cy + sn * corners[:, 0] + cs * corners[:, 1]], 1)
                c, r, v = self.cam.project_ground(pts)
                for a0, b0 in G.runs(v):
                    if b0 > a0:
                        d.line(sc(np.stack([c[a0:b0 + 1], r[a0:b0 + 1]], 1)), fill=AGENT + (230,), width=2)
        hj = self.hidx(i)
        if hj is not None:                                              # refcv8 BOX head footprints (orange), presence >= box_thr
            for corners, pp in self.head_boxes(hj, i, raw):
                if not (3.0 < corners[:, 0].mean() < 75.0):
                    continue
                c, r, v = self.cam.project_ground(corners)
                for a0, b0 in G.runs(v):
                    if b0 > a0:
                        d.line(sc(np.stack([c[a0:b0 + 1], r[a0:b0 + 1]], 1)), fill=(255, 170, 60, int(120 + 135 * min(pp * 2, 1.0))), width=3)
        # GT ribbon + line
        gt = self.gt_future(raw)
        dn = G.densify(gt, step=0.5) if len(gt) else np.zeros((0, 2))
        if len(dn) >= 2:                                                  # a stationary ego leaves a 1-point path: nothing to ribbon
            nrm = np.gradient(dn, axis=0)
            nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
            nrm = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
            Lp, Rp = dn + 0.9 * nrm, dn - 0.9 * nrm
            cl, rl, vl = self.cam.project_ground(Lp)
            cr, rr, vr = self.cam.project_ground(Rp)
            for k in range(len(dn) - 1):
                if vl[k] and vr[k] and vl[k + 1] and vr[k + 1]:
                    d.polygon(sc([(cl[k], rl[k]), (cl[k + 1], rl[k + 1]), (cr[k + 1], rr[k + 1]), (cr[k], rr[k])]), fill=GT + (42,))
            for poly in self.cam_xy(gt):
                d.line(sc(poly), fill=GT + (255,), width=4)
        # plan layers (held snapshot, re-expressed in the current ego frame)
        if i is not None:
            e = self.e
            for name, P8, col, wid, dash in (("prior", e.get("prior8"), PRIOR, 2, False), ("r7", None, R7, 3, True),
                                              ("ship", e.get("ship8"), SHIP, 3, False), ("r8", e["plan8"], R8, 4, False)):
                if name == "r7":
                    k = np.where(e["s6_ws"] == self.ws[i])[0]
                    P8 = e["s6_r7"][k[0]] if len(k) else e["r7_4"][i]
                if P8 is None:
                    continue
                P = self.held_plan(i, raw, P8[i] if (hasattr(P8, "ndim") and P8.ndim == 3) else P8)
                for poly in self.cam_xy(P):
                    pp = sc(poly)
                    if dash:
                        dashed(d, pp, col + (255,), wid)
                    else:
                        d.line(pp, fill=col + (255,), width=wid)
                if name == "r8":                                          # waypoint dots + time labels
                    c, r, v = self.cam.project_ground(P)
                    for k, (cc, rr_, ok) in enumerate(zip(c, r, v)):
                        if ok:
                            px, py = cc * s, rr_ * s
                            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=R8 + (255,), outline=(0, 0, 0, 255))
                            if D.SLOT_S8[k] in (2.0, 4.0, 6.0):
                                txt(d, (px + 8, py - 8), f"{D.SLOT_S8[k]:.0f}s", F_S, R8 + (255,))
        img.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"), (x0, y0))
        # legend + status (drawn straight on the frame)
        dd = ImageDraw.Draw(img, "RGBA")
        dd.rectangle([x0, y0, x0 + w, y0 + 30], fill=(0, 0, 0, 170))
        lx = x0 + 10
        for name, col, dash_ in (("GT path", GT, False), (f"{self.B.name_label} plan", R8, False), (f"shipped re-timed ({self.B.deploy_label} layer)", SHIP, False),
                                 ("refcv7-50,400", R7, True), ("causal prior", PRIOR, False), ("GT agents (z=0)", AGENT, False)):
            if dash_:
                dashed(dd, [(lx, y0 + 15), (lx + 26, y0 + 15)], col, 3, dash=7, gap=5)
            else:
                dd.line([(lx, y0 + 15), (lx + 26, y0 + 15)], fill=col, width=4)
            txt(dd, (lx + 31, y0 + 14), name, F_S, FG, "lm")
            lx += 31 + int(dd.textlength(name, font=F_S)) + 16
        dd.rectangle([x0, y0 + h - 26, x0 + w, y0 + h], fill=(0, 0, 0, 170))
        stat = (f"FRONT CAMERA  camera_front_wide_120fov  CYLINDRICAL 416x1024 f_ref {D.FRAME_F_REF:.1f}  (column linear in azimuth)   "
                f"raw frame {raw}/{self.R - 1}  t={raw * D.DT:4.1f} s")
        txt(dd, (x0 + 10, y0 + h - 13), stat, F_XS, DIM, "lm")
        if self.B.rt_note:
            dd.rectangle([x0, y0 + 30, x0 + w, y0 + 52], fill=(70, 20, 20, 200))
            txt(dd, (x0 + 10, y0 + 41), self.B.rt_note + "  (cyan shipped-plan layer not drawn; causal prior comes from the same file)", F_S, (255, 220, 220), "lm")
        if i is None:
            dd.rectangle([x0 + w // 2 - 280, y0 + h // 2 - 20, x0 + w // 2 + 280, y0 + h // 2 + 20], fill=(0, 0, 0, 190))
            msg = "no plan window at this frame (warm-up)" if raw - D.K < self.ws[0] else "no plan window at this frame (past the last window)"
            txt(dd, (x0 + w // 2, y0 + h // 2), msg, F_H, FG, "mm")
        else:
            txt(dd, (x0 + w - 10, y0 + h - 13), f"plan snapshot {i + 1}/{self.n} (2 Hz), age {age * D.DT:.1f} s, held + re-framed", F_XS, DIM, "rm")

    def _map_rgb(self, raw):
        if self.map_frac is None:
            return None
        fr = self.map_frac[raw].astype(np.float64) / 255.0                # [9, 200, 120]
        rgb = np.einsum("cxy,ck->xyk", fr, MAP_COL)                       # [200, 120, 3]  (x fwd, y: col0 = RIGHT)
        return np.clip(rgb[::-1, ::-1], 0, 255).astype(np.uint8)          # up = forward, right = RIGHT

    def draw_bev(self, img, raw, i, age):
        xg, yg, w, h = BEV_RECT
        x0, y0 = 0, 0
        canvas = Image.new("RGB", (w, h), PANEL)                          # panel-local canvas: nothing can overflow into a neighbour
        dd = ImageDraw.Draw(canvas, "RGBA")
        dd.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=PANEL, outline=EDGE)
        px = 8.0                                                           # px per metre
        cx0, cy0 = x0 + w / 2.0, y0 + 28 + 60 * px                         # ego (rig origin) at bottom centre
        # map underlay (dim)
        mrgb = self._map_rgb(raw)
        if mrgb is not None:
            # after the [::-1, ::-1] flip: rows 0..199 are x = 100..0 (top->bottom), cols 0..119 are y = +30..-30 (left->right)
            sub = mrgb[200 - 120:200, 20:100]                              # x 0..60 m, y +20..-20
            im = Image.fromarray((sub * 0.55).astype(np.uint8)).resize((int(80 * 4), int(120 * 4)), Image.NEAREST)
            canvas.paste(im, (int(cx0 - 20 * px), int(y0 + 28)))
        else:
            dd.rectangle([cx0 - 20 * px, y0 + 28, cx0 + 20 * px, y0 + 28 + 60 * px], fill=(30, 33, 40))
            txt(dd, (cx0, y0 + 60), "no map GT for this clip", F_S, DIM, "mm")

        def P(xy):  # ego (x fwd, y left) -> px
            a = np.asarray(xy, np.float64)
            return [(cx0 - yy * px, cy0 - xx * px) for xx, yy in a]
        for m in range(10, 61, 10):                                        # range rings / grid
            dd.line([(cx0 - 20 * px, cy0 - m * px), (cx0 + 20 * px, cy0 - m * px)], fill=(255, 255, 255, 40), width=1)
            txt(dd, (cx0 - 20 * px + 3, cy0 - m * px - 12), f"{m}m", F_XS, (255, 255, 255, 120))
        dd.line([(cx0, cy0), (cx0, cy0 - 60 * px)], fill=(255, 255, 255, 40), width=1)
        # GT agents
        ag = self.B.agents_at(self.ci, raw)
        if ag is not None:
            for a in ag[0]:
                cx, cy, yaw, l, wd, occ = a[:6]
                if not (-3 < cx < 62 and abs(cy) < 22):
                    continue
                cs, sn = math.cos(yaw), math.sin(yaw)
                co = np.array([[l / 2, wd / 2], [l / 2, -wd / 2], [-l / 2, -wd / 2], [-l / 2, wd / 2]])
                pts = P(np.stack([cx + cs * co[:, 0] - sn * co[:, 1], cy + sn * co[:, 0] + cs * co[:, 1]], 1))
                dd.polygon(pts, fill=AGENT + ((40,) if occ > 0.5 else (120,)), outline=AGENT + ((120,) if occ > 0.5 else (255,)))
        hj = self.hidx(i)
        if hj is not None:
            for corners, pp in self.head_boxes(hj, i, raw):
                if -3 < corners[:, 0].mean() < 62 and abs(corners[:, 1].mean()) < 22:
                    dd.line(P(corners), fill=(255, 170, 60, 255), width=2)
        # ego car (rear axle at origin; ~4.8 x 1.9 m)
        dd.polygon(P([[3.8, 0.95], [3.8, -0.95], [-1.0, -0.95], [-1.0, 0.95]]), fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))
        # trajectories
        gt = self.gt_future(raw)
        if len(gt):
            dd.line(P(np.vstack([[0, 0], gt])), fill=GT + (255,), width=4)
        if i is not None:
            e = self.e
            kk = np.where(e["s6_ws"] == self.ws[i])[0]
            layers = [("prior", e.get("prior8"), PRIOR, 2, False), ("r7", None, R7, 3, True), ("ship", e.get("ship8"), SHIP, 3, False), ("r8", e["plan8"], R8, 4, False)]
            for name, P8, col, wid, dash in layers:
                if name == "r7":
                    seq = e["s6_r7"][kk[0]] if len(kk) else e["r7_4"][i]
                elif P8 is None:
                    continue
                else:
                    seq = P8[i]
                pts = P(np.vstack([[0, 0], self.held_plan(i, raw, seq)]))
                if dash:
                    dashed(dd, pts, col + (255,), wid)
                else:
                    dd.line(pts, fill=col + (255,), width=wid)
                if name == "r8":
                    for k, p in enumerate(pts[1:]):
                        dd.ellipse([p[0] - 3.5, p[1] - 3.5, p[0] + 3.5, p[1] + 3.5], fill=R8 + (255,), outline=(0, 0, 0, 255))
        dd.rectangle([x0 + 1, y0 + 1, x0 + w - 2, y0 + 27], fill=PANEL)                  # header strip last: trajectories cannot cross it
        txt(dd, (x0 + 8, y0 + 6), "BEV  0-60 m ahead, +-20 m", F_H)
        dd.rectangle([x0 + 1, y0 + h - 17, x0 + w - 2, y0 + h - 1], fill=(0, 0, 0, 175))   # band so the footer never fights the ego marker / plan line
        txt(dd, (x0 + w - 4, y0 + h - 5), "blue = GT agents" + ("; orange = box head p>=%.2f" % self.box_thr if self.heads is not None else ""), F_XS, (200, 205, 215), "rs")
        img.paste(canvas, (xg, yg))

    def draw_map(self, img, raw, i, age):
        xg, yg, w, h = MAP_RECT
        x0, y0 = 0, 0
        canvas = Image.new("RGB", (w, h), PANEL)
        dd = ImageDraw.Draw(canvas, "RGBA")
        dd.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=PANEL, outline=EDGE)
        mrgb = self._map_rgb(raw)
        hj = self.hidx(i)
        has_head = hj is not None and self.heads is not None and "map_class" in self.heads
        if has_head:                                                     # two columns, +-20 m, 4.2 px/m: GT | refcv8 map head
            PS, CW, CH = 4.2, 168, 420
            cols = [(x0 + 2, "GT (SAM3)"), (x0 + 2 + CW + 2, "refcv8 head")]
            for k, (cx0, lab) in enumerate(cols):
                if k == 0:
                    if mrgb is None:
                        txt(dd, (cx0 + CW / 2, y0 + 32 + CH / 2), "no map GT", F_S, DIM, "mm")
                        im = None
                    else:
                        im = Image.fromarray(mrgb[:, 20:100]).resize((CW, CH), Image.NEAREST)          # GT: rows x 100..0, cols y +20..-20
                else:
                    code = self.heads["map_class"][hj][::-1, ::-1][:, 100:500]                           # head: 0.1 m, flipped like the GT -> y +20..-20
                    im = Image.fromarray(MAP_COL[np.clip(code, 0, 7)].astype(np.uint8)).resize((CW, CH), Image.NEAREST)
                if im is not None:
                    canvas.paste(im, (cx0, y0 + 32))
                ex, ey = cx0 + CW / 2.0, y0 + 32 + CH

                def P(xy, ex=ex, ey=ey):
                    return [(ex - yy * PS, ey - xx * PS) for xx, yy in np.asarray(xy, np.float64)]
                gt = self.gt_future(raw)
                if len(gt):
                    dd.line(P(np.vstack([[0, 0], gt])), fill=GT + (255,), width=2)
                if i is not None:
                    dd.line(P(np.vstack([[0, 0], self.held_plan(i, raw, self.e["plan8"][i])])), fill=R8 + (255,), width=2)
                dd.polygon(P([[3.8, 0.95], [3.8, -0.95], [-1.0, -0.95], [-1.0, 0.95]]), fill=(240, 240, 240, 220))
                txt(dd, (cx0 + 3, y0 + 33), lab, F_XS, (255, 255, 255, 230))
            ly = y0 + 32 + CH + 5
            note = f"head: snapshot {age * D.DT:.1f} s old; rule prior_corrected"
        else:
            MS = 2.1                                                        # px per 0.5 m cell
            ps = MS * 2.0                                                   # px per metre
            mx, my = x0 + (w - 120 * MS) / 2.0, y0 + 28
            if mrgb is None:
                txt(dd, (x0 + w / 2, y0 + h / 2 - 20), "no map GT for this clip", F_N, DIM, "mm")
            else:
                im = Image.fromarray(mrgb).resize((int(120 * MS), int(200 * MS)), Image.NEAREST)
                canvas.paste(im, (int(mx), int(my)))
                ex, ey = mx + 60 * MS, my + 200 * MS                          # ego at bottom centre (y = 0 -> col 60 after flip)

                def P(xy):
                    return [(ex - yy * ps, ey - xx * ps) for xx, yy in np.asarray(xy, np.float64)]
                gt = self.gt_future(raw)
                if len(gt):
                    dd.line(P(np.vstack([[0, 0], gt])), fill=GT + (255,), width=3)
                if i is not None:
                    dd.line(P(np.vstack([[0, 0], self.held_plan(i, raw, self.e["plan8"][i])])), fill=R8 + (255,), width=3)
                dd.polygon(P([[3.8, 0.95], [3.8, -0.95], [-1.0, -0.95], [-1.0, 0.95]]), fill=(240, 240, 240, 220))
            ly = my + 200 * MS + 6
            note = None
        lx = x0 + 6
        for k in (1, 2, 3, 4, 5, 6, 7, 8):
            if has_head and k == 8:
                continue
            nm = "drive" if k == 1 else MAP_NAMES[k]                         # short form so the legend fits the 340 px panel on ONE row (it was clipped at 'wa')
            if lx + 12 + int(dd.textlength(nm, font=F_XS)) > x0 + w - 3:     # wrap rather than clip: a clipped legend hides classes
                lx = x0 + 6
                ly += 13
            dd.rectangle([lx, ly + 2, lx + 9, ly + 11], fill=tuple(int(v) for v in MAP_COL[k]), outline=(120, 120, 120))
            txt(dd, (lx + 12, ly), nm, F_XS, DIM)
            lx += 12 + int(dd.textlength(nm, font=F_XS)) + 5
        if has_head:
            txt(dd, (x0 + 6, ly + 17), note, F_XS, (255, 190, 80))
        else:
            txt(dd, (x0 + 6, ly + 17), "refcv8 MAP head: PENDING dump (dump_heads_r8.py)", F_XS, (255, 190, 80))
        dd.rectangle([x0 + 1, y0 + 1, x0 + w - 2, y0 + 27], fill=PANEL)
        txt(dd, (x0 + 8, y0 + 6), "MAP  GT vs refcv8 head, 100 x 40 m" if has_head else "MAP  GT (SAM3 v3), 100 x 60 m", F_H)
        img.paste(canvas, (xg, yg))

    def _alloc_text(self, i):
        hj = self.hidx(i)
        if hj is None or "out_r8_hyp_alloc" not in self.heads:
            return "n/a (pending GPU dump)"
        H = self.heads
        hyp = H["out_r8_hyp_alloc"][hj].astype(int)
        lat3 = H["out_r8_lat3"][hj][H["out_r8_alloc"][hj]] if "out_r8_lat3" in H else np.array([], int)
        lon6 = H["out_r8_lon6"][hj][H["out_r8_alloc"][hj]] if "out_r8_lon6" in H else np.array([], int)
        return (f"{len(hyp)} hyps; lat3 {''.join(str(v) + ' ' for v in np.bincount(lat3, minlength=3))}| lon6 {''.join(str(v) + ' ' for v in np.bincount(lon6, minlength=6))}"
                if len(lat3) else f"{len(hyp)} hyps")

    def draw_tactical(self, img, raw, i, age):
        xg, yg, w, h = TAC_RECT
        x0, y0 = 0, 0
        canvas = Image.new("RGB", (w, h), PANEL)
        dd = ImageDraw.Draw(canvas, "RGBA")
        dd.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=PANEL, outline=EDGE)
        txt(dd, (x0 + 10, y0 + 6), "TACTICAL / STRATEGIC / SPEED  (refcv8, held snapshot)", F_H)
        if i is None:
            txt(dd, (x0 + w / 2, y0 + h / 2), "no plan window at this frame", F_H, DIM, "mm")
            img.paste(canvas, (xg, yg))
            return
        e = self.e

        def bars(y, title, names, logits, gt_idx):
            pr = D.softmax(logits)
            am = int(pr.argmax())
            gtn = "unlabelled" if gt_idx < 0 else names[gt_idx]
            ok = "" if gt_idx < 0 else ("  OK" if am == gt_idx else "  MISS")
            txt(dd, (x0 + 10, y), f"{title}  pred: {names[am]}   GT: {gtn}{ok}", F_N, (FG if gt_idx < 0 else (GOOD if am == gt_idx else BAD)))
            for k, nm in enumerate(names):
                yy = y + 21 + k * 15
                txt(dd, (x0 + 14, yy), nm, F_SM, FG if k == am else DIM)
                bx = x0 + 215
                dd.rectangle([bx, yy + 1, bx + 330, yy + 12], outline=EDGE)
                dd.rectangle([bx, yy + 1, bx + int(330 * pr[k]), yy + 12], fill=(R8 if k == am else (110, 74, 20)))
                txt(dd, (bx + 336, yy), f"{100 * pr[k]:4.0f}%", F_SM, DIM)
                if k == gt_idx:
                    dd.polygon([(bx - 12, yy + 1), (bx - 12, yy + 12), (bx - 3, yy + 6)], fill=GT)
            return y + 21 + len(names) * 15 + 3
        y = y0 + 32
        y = bars(y, "LAT", D.LAT_NAMES, e["lat_logits"][i], int(e["lat_label"][i]))
        y = bars(y + 1, "LON", D.LON_NAMES, e["lon_logits"][i], int(e["lon_label"][i]))
        txt(dd, (x0 + 10, y), "bars = softmax of the v6 behaviour decoder (tacv6_lat/lon_logits); green triangle = GT label", F_XS, DIM)
        y += 14
        v0 = float(e["v0"][i])
        vm = float(e["v_max_ms"][i])
        nav_fed, nav_clip = FEED.label(int(e["nav_fed"][i]), int(e["nav_clip"][i]), bool(e["r8_nav_known"][i]), e["nav_feed"])
        fb, gcls, pcls, vend = self.fb[i]
        row = [("NAV FED (this frame)", f"{nav_fed}{'' if e['nav_valid'][i] else ' (invalid)'}"),
               ("clip nav (not fed)" if e["nav_feed"] == FEED.NAV_FEED_PF else "clip nav", f"{nav_clip}   |   STRATEGIC: off (--no-strategic)"),
               ("ALLOCATION (head dump)", self._alloc_text(i)),
               ("speed (actual v0)", f"{v0:5.1f} m/s = {3.6 * v0:4.0f} km/h"),
               ("fed ceiling (road-type)", f"{vm:5.1f} m/s = {3.6 * vm:4.0f} km/h"),
               ("plan end-speed (5-6 s)", f"{vend:5.1f} m/s   (dv vs v0 {vend - v0:+5.1f})")]
        if gcls is not None:
            row.append(("GT speed @ +6 s", f"{e['s6_gt_v6'][list(e['s6_ws']).index(int(self.ws[i]))]:5.1f} m/s   GT class: {gcls}"))
        else:
            row.append(("GT speed @ +6 s", "n/a (window runs past the clip end)"))
        for k, (a, b) in enumerate(row):
            txt(dd, (x0 + 14, y + k * 16), a, F_SM, DIM)              # 16 px (was 17): the 7th row (clip nav, 2026-10-20) must fit the panel
            txt(dd, (x0 + 214, y + k * 16), b, F_SM, FG)
        y += len(row) * 16 + 4
        # false-brake banner
        if fb is None:
            txt(dd, (x0 + 10, y + 4), "FALSE-BRAKE: n/a (no GT 6 s for this window)", F_H, DIM)
        elif fb:
            dd.rectangle([x0 + 8, y, x0 + w - 8, y + 30], fill=(120, 20, 20, 230), outline=BAD)
            txt(dd, (x0 + 16, y + 15), f"FALSE BRAKE: plan decelerates ({vend - v0:+.1f} m/s) while GT cruises", F_H, (255, 220, 220), "lm")
        else:
            txt(dd, (x0 + 10, y + 4), f"no false-brake (GT class {gcls}, plan class {pcls})", F_H, GOOD)
        txt(dd, (x0 + 10, y + 34), "false-brake = GT 'cruising' (|v6-v0|<1.5 m/s) AND plan 'decel' (last-second speed - v0 <= -1.5)", F_XS, DIM)
        txt(dd, (x0 + 10, y + 47), "definition: cruise_brake_retime_r8.py / s6_lon_decomp.py (battery package)", F_XS, DIM)
        img.paste(canvas, (xg, yg))

    def draw_metrics(self, img, raw, i, age):
        xg, yg, w, h = MET_RECT
        x0, y0 = 0, 0
        canvas = Image.new("RGB", (w, h), PANEL)
        dd = ImageDraw.Draw(canvas, "RGBA")
        dd.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=PANEL, outline=EDGE)
        txt(dd, (x0 + 10, y0 + 6), "METRICS  (ADE vs the logged GT path; this window | clip mean so far)", F_H)
        cols = [("ADE 2 s", "this", "so far"), ("ADE 6 s", "this", "so far")]
        tx = x0 + 12
        txt(dd, (tx + 232, y0 + 34), "ADE 2 s (0.5-2 s)", F_S, FG)
        txt(dd, (tx + 232 + 188, y0 + 34), "ADE 6 s (1-6 s)", F_S, FG)
        for k, t in enumerate(("window", "so far", "window", "so far")):
            txt(dd, (tx + 232 + (k % 2) * 88 + (k // 2) * 188, y0 + 52), t, F_XS, DIM)
        rows = [(self.B.name_label, "r8", R8, False), ("refcv7-50,400", "r7", R7, False), (f"shipped re-timed ({self.B.deploy_label})*", "ship", SHIP, False), ("causal prior", "prior", PRIOR, False)]
        for k, (lab, key, col, _) in enumerate(rows):
            yy = y0 + 72 + k * 24
            dd.rectangle([tx, yy + 3, tx + 14, yy + 15], fill=col)
            avail = 232 - 22 - 8                                             # the label must end before the first number column
            fnt = F_N if dd.textlength(lab, font=F_N) <= avail else F_S       # the 'shipped re-timed (...)' label overflowed onto the numbers
            if dd.textlength(lab, font=fnt) > avail:                          # still too wide: cut and mark the cut, never overlap
                while len(lab) > 4 and dd.textlength(lab + "...", font=fnt) > avail:
                    lab = lab[:-1]
                lab = lab.rstrip() + "..."
            txt(dd, (tx + 22, yy), lab, fnt, FG)
            if key not in self.ade:
                continue
            for j, (m, kind) in enumerate((("a2", 0), ("a2", 1), ("a6", 0), ("a6", 1))):
                if i is None:
                    s = "-"
                else:
                    v = self.ade[key][m][i] if kind == 0 else self.run[key][m][i]
                    s = "n/a" if not np.isfinite(v) else f"{v:5.2f}"
                txt(dd, (tx + 232 + (j % 2) * 88 + (j // 2) * 188, yy), s, F_NM, FG)
        for k, ln in enumerate(("ADE = mean over slots of the Euclidean error (m) vs the logged GT path.",
                                "6-s cells exist only for windows whose 6 s GT lies inside the clip.",
                                f"* shipped re-timed = RETIME_DEPLOY8 on the {self.B.deploy_label} ckpt (the banked",
                                f"  deploy layer; NOT this checkpoint's plan unless equal).  refcv7 = banked paired arm.",
                                "T1 open loop on LOGGED frames: the plan never steers the camera.")):
            txt(dd, (tx, y0 + 170 + k * 14), ln, F_XS, DIM)
        # clip-level mean ADE (final) box
        fin2 = {k: np.nanmean(v["a2"]) for k, v in self.ade.items()}
        fin6 = {k: (np.nanmean(v["a6"]) if np.isfinite(v["a6"]).any() else np.nan) for k, v in self.ade.items()}
        txt(dd, (tx, y0 + 250), "this clip, all windows:", F_S, FG)
        for k, key in enumerate(("r8", "r7", "ship", "prior")):
            if key in fin2:
                txt(dd, (tx + 14, y0 + 270 + k * 17), f"{key:5s} ADE2 {fin2[key]:5.2f} m   ADE6 {fin6[key]:5.2f} m", F_SM, DIM)
        # nav / label summary box
        self._speed_plot(dd, (x0 + 590, y0 + 28, w - 600, h - 40), raw, i)
        img.paste(canvas, (xg, yg))

    def _speed_plot(self, dd, rect, raw, i):
        x, y, w, h = rect
        dd.rectangle([x, y, x + w, y + h], fill=(18, 20, 25), outline=EDGE)
        t_end = (self.R - 1) * D.DT
        vmax = max(float(self.raw_poses[:, 3].max()) * 1.25, 12.0)
        L, Rr, T, Bt = x + 46, x + w - 12, y + 24, y + h - 46

        def X(t):
            return L + (Rr - L) * t / t_end

        def Y(v):
            return Bt - (Bt - T) * min(max(v, 0.0), vmax) / vmax
        txt(dd, (x + 8, y + 5), "SPEED (m/s) vs TIME: plan vs logged human speed", F_S, FG)
        for v in range(0, int(vmax) + 1, 5 if vmax > 18 else 2):
            dd.line([(L, Y(v)), (Rr, Y(v))], fill=(255, 255, 255, 25))
            txt(dd, (L - 5, Y(v)), f"{v}", F_XS, DIM, "rm")
        for t in range(0, int(t_end) + 1, 2):
            dd.line([(X(t), T), (X(t), Bt)], fill=(255, 255, 255, 18))
            txt(dd, (X(t), Bt + 4), f"{t}s", F_XS, DIM, "ma")
        # logged human speed (the poses' v column)
        vt = self.raw_poses[:, 3]
        dd.line([(X(k * D.DT), Y(float(v))) for k, v in enumerate(vt)], fill=GT + (255,), width=3)
        if i is not None:
            e = self.e
            # first-slot plan speed of every window so far (what each snapshot said to do NEXT)
            for j in range(i + 1):
                s0 = D.plan_speeds(e["plan8"][j])[0]
                tt = (int(self.ws[j]) + D.K) * D.DT + 0.25
                dd.ellipse([X(tt) - 2.5, Y(s0) - 2.5, X(tt) + 2.5, Y(s0) + 2.5], fill=R8 + (150,))
            t0 = (int(self.ws[i]) + D.K) * D.DT
            edges = np.concatenate([[0.0], D.SLOT_S8])
            for nm, P8, col, dsh in (("prior", e.get("prior8"), PRIOR, False), ("ship", e.get("ship8"), SHIP, True), ("r8", e["plan8"], R8, False)):
                if P8 is None:
                    continue
                sp = D.plan_speeds(P8[i])
                pts = []
                for k in range(8):
                    pts += [(X(t0 + edges[k]), Y(sp[k])), (X(t0 + edges[k + 1]), Y(sp[k]))]
                (dashed(dd, pts, col + (255,), 2) if dsh else dd.line(pts, fill=col + (255,), width=3 if nm == "r8" else 2))
            vm = float(e["v_max_ms"][i])
            dashed(dd, [(X(t0), Y(vm)), (X(min(t0 + 6.0, t_end)), Y(vm))], CEIL + (255,), 2, dash=4, gap=4)
            txt(dd, (X(min(t0 + 6.0, t_end)) - 3, Y(vm) + (-12 if vm <= vmax else 2)), f"ceiling {3.6 * vm:.0f} km/h" + ("" if vm <= vmax else " (off scale)"), F_XS, CEIL, "ra")
        nowx = X(raw * D.DT)
        dd.line([(nowx, T), (nowx, Bt)], fill=(255, 255, 255, 200), width=1)
        lx = L + 6
        for nm, col in (("logged speed", GT), ("refcv8 plan (held 6 s)", R8), (f"shipped ({self.B.deploy_label})", SHIP), ("prior", PRIOR), ("fed ceiling", CEIL)):
            txt(dd, (lx, y + h - 12), "- " + nm, F_XS, col, "lm")
            lx += int(dd.textlength("- " + nm, font=F_XS)) + 14

    def draw_title(self, img, raw):
        dd = ImageDraw.Draw(img, "RGBA")
        dd.rectangle([0, 0, W, TITLE_H], fill=(8, 10, 14))
        step = self.B.ckpt_step
        md5 = (self.B.ckpt_md5 or "")[:8]
        txt(dd, (12, 8), f"refcv8-r101-s0   ckpt: {'EMA' if self.B.ema else 'raw'} step {step:,} ({'ckpt_ema_' if self.B.ema else 'ckpt_'}{step}.pt md5 {md5})   arm: refcv8 hierarchical, --no-strategic   "
                         f"episode {self.ci}", F_TITLE)
        feed = ("nav feed: PER-FRAME v9 token + r8_nav (as trained)" if self.e["nav_feed"] == FEED.NAV_FEED_PF
                else "nav feed: CLIP-level token, no r8_nav (NOT the trained feed)")
        txt(dd, (12, 31), "TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)   |   10 Hz native, plan snapshots 2 Hz   |   "
                          f"inference seed 0, single training seed   |   {feed}", F_S, DIM)
        txt(dd, (W - 12, 8), f"{raw * D.DT:5.1f} s", F_TITLE, FG, "ra")

    def render(self, raw):
        img = Image.new("RGB", (W, H), BG)
        i, age = self.held(raw)
        self.draw_camera(img, raw, i, age)
        self.draw_tactical(img, raw, i, age)
        self.draw_bev(img, raw, i, age)
        self.draw_map(img, raw, i, age)
        self.draw_metrics(img, raw, i, age)
        self.draw_title(img, raw)
        return img


# --------------------------------------------------------------------------------------------- video IO
def open_ffmpeg(out, fps=10, crf=18, threads=4):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads), "-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def title_card(lines, sub=None):
    img = Image.new("RGB", (W, H), (8, 10, 14))
    d = ImageDraw.Draw(img)
    y = H // 2 - 80
    for k, ln in enumerate(lines):
        d.text((W // 2, y + k * 58), ln, font=font(44 if k == 0 else 26, bold=(k == 0)), fill=FG if k == 0 else DIM, anchor="mm")
    if sub:
        for k, ln in enumerate(sub):
            d.text((W // 2, y + 190 + k * 30), ln, font=font(22), fill=DIM, anchor="mm")
    return img


def write_card(out, img, seconds=3.0, fps=10, crf=18):
    p = open_ffmpeg(out, fps, crf)
    raw = np.asarray(img, np.uint8).tobytes()
    for _ in range(int(seconds * fps)):
        p.stdin.write(raw)
    p.stdin.close()
    return p.wait()


def render_episode(bank, ci, out, tree_label, max_frames=None, fps=10, crf=18, still=None, png=None, progress=True, heads_dir=None, box_p_thr=0.20):
    t0 = time.time()
    ep = Episode(bank, ci, tree_label, heads_dir=heads_dir, box_p_thr=box_p_thr)
    if still is not None:
        im = ep.render(int(still))
        im.save(png)
        return {"png": png, "controls": ep.ctl}
    nfr = ep.R if max_frames is None else min(ep.R, int(max_frames))
    pr = open_ffmpeg(out, fps, crf)
    for raw in range(nfr):
        im = ep.render(raw)
        pr.stdin.write(np.asarray(im, np.uint8).tobytes())
        if progress and raw % 25 == 0:
            print(f"  ep{ci} frame {raw}/{nfr}  {time.time() - t0:.0f}s", flush=True)
    pr.stdin.close()
    rc = pr.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")
    return {"out": out, "frames": nfr, "seconds": nfr / fps, "wall_s": time.time() - t0, "controls": ep.ctl,
            "map_gt": ep.map_frac is not None, "windows": ep.n, "nav_feed": ep.e["nav_feed"],
            "n_windows_fed_token_differs_from_clip": int((ep.e["nav_fed"] != ep.e["nav_clip"]).sum()),
            "rt": bank.rt_note or f"ON ({bank.deploy_label})", "heads_feed": (None if ep.heads is None else str(ep.heads.get("nav_feed", "clip (legacy)")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="step40000ema_unc")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--still", type=int, default=None, help="render ONE raw frame to --png instead of a video")
    ap.add_argument("--png", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--deploy-npz", default=D.AUTO, help="DEPLOY8_POS_*.npz (shipped re-timed plan + causal prior). auto (DEFAULT): per-clip tree -> the 30k EMA "
                    "table, labelled; per-frame (*_pf) tree -> the REGISTERED per-frame table's npz, or the RT layer is OFF with an on-screen note")
    ap.add_argument("--arch-root", default=str(D.ARCH))
    ap.add_argument("--pf-registered", default=str(D.PF_REGISTERED))
    ap.add_argument("--pf-pkg", default=str(D.PF_PKG))
    ap.add_argument("--heads-dir", default=None, help="dir with heads_ep###.npz from dump_heads_r8.py (fills the MAP-head / BOX-head / allocation layers)")
    ap.add_argument("--box-p-thr", type=float, default=0.20, help="box-head presence threshold for display (the head never exceeds p 0.40 on the three showcase episodes, MEASURED: a standard 0.5 shows nothing; this is a display choice, not a detector threshold)")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    B = D.EvalBank(a.tree, deploy_npz=a.deploy_npz, arch_root=a.arch_root, pf_registered=a.pf_registered, pf_pkg=a.pf_pkg)
    res = render_episode(B, a.episode, a.out, a.tree, a.max_frames, crf=a.crf, still=a.still, png=a.png, heads_dir=a.heads_dir, box_p_thr=a.box_p_thr)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
