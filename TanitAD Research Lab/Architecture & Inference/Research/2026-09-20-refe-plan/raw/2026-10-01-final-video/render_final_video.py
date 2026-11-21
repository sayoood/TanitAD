"""Render REFe's final-model planning behaviour as a long video: for each navtest scene, the 64 trajectory hypotheses
(coloured by the scorer's predicted PDMS), the selected plan, the best plan in hindsight and the human future, overlaid
on the FRONT camera and on a metric BEV, with the NAVSIM driving command, the other three cameras and the harness's
true PDMS for the selected / best / mean hypothesis.

Inputs (all MEASURED artifacts of the 200-token navtest subset, model_final):
  proptable/sub200_final/proposals.npz  -> proposals (200,64,8,3) on NAVSIM's 0.5 s grid in the ego frame, scorer logits, pick
  proptable/sub200_final/table.npz      -> the harness's TRUE PDMS and sub-scores of every proposal
  W3 export                             -> human future, driving command, log of every token
  navsim_logs/test/<log>.pkl            -> camera calibration, current-frame image names, logged agent boxes
  frames/<log>_<CAM>.zip                -> the camera images

    python render_final_video.py --out final_navtest_sub200.mp4 [--limit N] [--still K --still-out out.png]

Evidence class: the overlays are a rendering of banked numbers; nothing here is re-inferred. The BEV shows the LOGGED agents
(ground-truth annotations), which the model never sees. Rendering only: no model, no GPU.
"""
import argparse
import gzip
import json
import math
import os
import pickle
import subprocess
import sys
import zipfile

import cv2
import numpy as np

_DRV = os.environ.get("REFE_DRIVE", "D:")      # the external drive was re-lettered D: -> E: on 2026-10-03
DATA = f"{_DRV}/Projects/TanitAD/data/refe_navtest"
EXPORT = f"{_DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
LOGS = f"{_DRV}/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
FFMPEG = "C:/Users/Admin/venvs/tanitad/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
W, H = 1920, 1080
FRONT = (0, 0, 1280, 720)          # x, y, w, h of the front-camera panel
BEVP = (1280, 0, 640, 720)
BOTTOM_Y = 720
EGO_L, EGO_W, EGO_REAR = 4.8, 1.95, 1.0   # rear-axle origin: the car spans x in [-1.0, +3.8]
GROUND_Z = 0.0
FPS = 15
CLIP_S = 3.2
CMD = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def text(img, s, org, scale=0.6, col=(255, 255, 255), th=1, outline=True):
    if outline:
        cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


class Cam:
    """pinhole projection of ego-frame points (lidar frame == ego frame in NAVSIM: lidar2ego is the identity)"""

    def __init__(self, c, l2e_t, l2e_q):
        self.K = np.array(c["cam_intrinsic"], dtype=np.float64)
        self.R = np.array(c["sensor2lidar_rotation"], dtype=np.float64)
        self.t = np.array(c["sensor2lidar_translation"], dtype=np.float64)
        self.Rl = quat_to_mat(l2e_q)
        self.tl = np.asarray(l2e_t, dtype=np.float64)

    def project(self, P):
        """P: (N,3) ego frame -> (uv (N,2), z (N,)) in the ORIGINAL 1920x1080 image"""
        pl = (np.asarray(P, dtype=np.float64) - self.tl) @ self.Rl            # ego -> lidar
        pc = (pl - self.t) @ self.R                                            # lidar -> camera
        z = pc[:, 2]
        uv = (pc @ self.K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
        return uv, z


def densify(poses, n=48):
    """poses (8,3) on the 0.5 s grid -> a dense ego-frame path (n,2) starting at the rear axle, and its times"""
    xy = np.vstack([[0.0, 0.0], poses[:, :2]])
    t = np.arange(len(xy)) * 0.5
    tt = np.linspace(0, t[-1], n)
    return np.stack([np.interp(tt, t, xy[:, 0]), np.interp(tt, t, xy[:, 1])], 1), tt


def ribbon(path, half_w):
    d = np.gradient(path, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return path + half_w * nrm, path - half_w * nrm


def draw_path_cam(img, cam, path, tt, tmax, col, th, scale, near=2.0):
    uv, z = cam.project(np.c_[path, np.full(len(path), GROUND_Z)])
    ok = (z > near) & (tt <= tmax + 1e-9)
    for i in range(len(path) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple((uv[i] * scale).astype(int)), tuple((uv[i + 1] * scale).astype(int)), col, th, cv2.LINE_AA)


def draw_ribbon_cam(overlay, cam, path, tt, tmax, col, scale, half_w=0.95, near=2.0):
    l, r = ribbon(path, half_w)
    ul, zl = cam.project(np.c_[l, np.full(len(l), GROUND_Z)])
    ur, zr = cam.project(np.c_[r, np.full(len(r), GROUND_Z)])
    ok = (zl > near) & (zr > near) & (tt <= tmax + 1e-9)
    for i in range(len(path) - 1):
        if ok[i] and ok[i + 1]:
            poly = np.array([ul[i], ul[i + 1], ur[i + 1], ur[i]]) * scale
            cv2.fillPoly(overlay, [poly.astype(np.int32)], col, cv2.LINE_AA)


def box_corners(cx, cy, yaw, l, w):
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


class Bev:
    def __init__(self, box, ppm=11.0, x0=-8.0):
        self.x, self.y, self.w, self.h = box
        self.ppm, self.x0 = ppm, x0

    def pt(self, x, y):
        return (int(self.x + self.w / 2 - y * self.ppm), int(self.y + self.h - (x - self.x0) * self.ppm))

    def poly(self, xy):
        return np.array([self.pt(a, b) for a, b in xy], dtype=np.int32)


def turbo(v):
    c = cv2.applyColorMap(np.uint8([[int(np.clip(v, 0, 1) * 255)]]), cv2.COLORMAP_TURBO)[0, 0]
    return tuple(int(x) for x in c)


def load_inputs():
    P = np.load(f"{DATA}/proptable/sub200_final/proposals.npz")
    T = np.load(f"{DATA}/proptable/sub200_final/table.npz", allow_pickle=True)
    assert [str(t) for t in P["token"]] == [str(t) for t in T["token"]]
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    return P, T, exp


def agg_score(logits):
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12      # navsim_v1: NC DAC (5EP 5TTC 2C)/12


class LogData:
    def __init__(self, log):
        self.log = log
        fr = pickle.load(open(f"{LOGS}/{log}.pkl", "rb"))
        self.by_token = {f["token"]: f for f in fr}
        self.zips = {}

    def image(self, name, cam):
        if cam not in self.zips:
            self.zips[cam] = zipfile.ZipFile(f"{DATA}/frames/{self.log}_{cam}.zip")
        buf = np.frombuffer(self.zips[cam].read(os.path.basename(name)), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fit(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def scene_base(ld, frame):
    canvas = np.full((H, W, 3), 18, np.uint8)
    imgs = {c: ld.image(frame["cams"][c]["data_path"], c) for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")}
    fx, fy, fw, fh = FRONT
    canvas[fy:fy + fh, fx:fx + fw] = fit(imgs["CAM_F0"], fw, fh)
    pw, ph = 480, 270
    for i, (c, label) in enumerate((("CAM_L0", "LEFT"), ("CAM_R0", "RIGHT"), ("CAM_B0", "REAR"))):
        canvas[BOTTOM_Y:BOTTOM_Y + ph, i * pw:(i + 1) * pw] = fit(imgs[c], pw, ph)
        text(canvas, f"{label} camera", (i * pw + 10, BOTTOM_Y + 24), 0.6)
    text(canvas, "FRONT camera", (14, 30), 0.8)
    return canvas


def render_scene(ld, tok, i, n, P, T, exp, run_stats, writer):
    frame = ld.by_token[tok]
    cam = Cam(frame["cams"]["CAM_F0"], frame["lidar2ego_translation"], frame["lidar2ego_rotation"])
    prop = P["proposals"][i].astype(np.float64)                  # (64,8,3)
    sc = agg_score(P["logits"][i])
    pick = int(P["pick"][i])
    pdms = T["pdms"][i]
    sub = T["sub"][i]
    valid = T["valid"][i].astype(bool)
    best = int(np.argmax(np.where(valid, pdms, -1)))
    hum = np.array(exp[tok]["human_future_poses"], dtype=np.float64)
    cmd = int(np.argmax(frame["driving_command"]))
    ego_v = float(np.hypot(*exp[tok]["ego_statuses"][-1]["ego_velocity"]))
    canvas0 = scene_base(ld, frame)
    bev = Bev(BEVP)
    bx, by, bw, bh = BEVP
    cv2.rectangle(canvas0, (bx, by), (bx + bw, by + bh), (28, 28, 30), -1)
    for r in range(10, 60, 10):
        cv2.line(canvas0, bev.pt(r, -29), bev.pt(r, 29), (50, 50, 54), 1)
        text(canvas0, f"{r} m", (bx + 6, bev.pt(r, 0)[1] - 3), 0.4, (120, 120, 125), 1, False)
    cv2.line(canvas0, bev.pt(-8, 0), bev.pt(57, 0), (50, 50, 54), 1)
    anns = frame["anns"]
    colors = {"vehicle": (220, 140, 60), "pedestrian": (60, 160, 255), "bicycle": (80, 220, 220), "generic_object": (150, 150, 150)}
    for b, nme in zip(np.array(anns["gt_boxes"]), anns["gt_names"]):
        if -10 < b[0] < 60 and abs(b[1]) < 30:
            cv2.fillPoly(canvas0, [bev.poly(box_corners(b[0], b[1], b[6], b[3], b[4]))], colors.get(str(nme), (150, 150, 150)), cv2.LINE_AA)
    text(canvas0, "BEV (ego frame, metric); logged agents are not a model input", (bx + 8, by + 24), 0.5, (200, 200, 205))
    ix, iy = 1440, 720
    cv2.rectangle(canvas0, (ix, iy), (W, H), (24, 24, 28), -1)
    cv2.rectangle(canvas0, (0, 990), (1440, H), (24, 24, 28), -1)
    paths = [densify(prop[k]) for k in range(64)]
    hpath = densify(hum)
    nfr = int(CLIP_S * FPS)
    order = np.argsort(sc)
    lo, hi = float(sc.min()), float(sc.max())
    nrm = (sc - lo) / max(hi - lo, 1e-9)
    tt_full = paths[0][1]
    for f in range(nfr):
        p = f / (nfr - 1)
        tmax = min(4.0, 4.0 * min(1.0, p / 0.45))                  # hypotheses grow over the first 45 %
        marker_t = 4.0 * p                                          # the ego marker moves over the whole clip
        canvas = canvas0.copy()
        fx, fy, fw, fh = FRONT
        sc_ = fw / 1920.0
        front = canvas[fy:fy + fh, fx:fx + fw]
        ov = front.copy()
        draw_ribbon_cam(ov, cam, paths[pick][0], tt_full, 4.0, (80, 220, 90), sc_)
        front[:] = cv2.addWeighted(ov, 0.30, front, 0.70, 0)
        for k in order:
            if k != pick:
                draw_path_cam(front, cam, paths[k][0], tt_full, tmax, turbo(0.15 + 0.85 * nrm[k]), 2, sc_)
        draw_path_cam(front, cam, hpath[0], hpath[1], tmax, (255, 255, 255), 3, sc_)
        if best != pick:
            draw_path_cam(front, cam, paths[best][0], tt_full, tmax, (255, 0, 255), 3, sc_)
        draw_path_cam(front, cam, paths[pick][0], tt_full, tmax, (0, 0, 0), 9, sc_)
        draw_path_cam(front, cam, paths[pick][0], tt_full, tmax, (90, 255, 90), 5, sc_)
        for j in range(8):
            tj = (j + 1) * 0.5
            if tj <= tmax + 1e-9:
                uv, z = cam.project(np.array([[prop[pick][j][0], prop[pick][j][1], GROUND_Z]]))
                if z[0] > 2.0:
                    c = tuple((uv[0] * sc_).astype(int))
                    cv2.circle(front, c, 7, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(front, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
                    if j % 2 == 1:
                        text(front, f"{tj:.0f}s", (c[0] + 9, c[1] - 6), 0.55, (255, 255, 255), 1)
        m = tt_full <= tmax + 1e-9
        for k in order:
            if k != pick:
                cv2.polylines(canvas, [bev.poly(paths[k][0][m])], False, turbo(0.15 + 0.85 * nrm[k]), 1, cv2.LINE_AA)
        mh = hpath[1] <= tmax + 1e-9
        cv2.polylines(canvas, [bev.poly(hpath[0][mh])], False, (255, 255, 255), 2, cv2.LINE_AA)
        if best != pick:
            cv2.polylines(canvas, [bev.poly(paths[best][0][m])], False, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [bev.poly(paths[pick][0][m])], False, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(canvas, [bev.poly(paths[pick][0][m])], False, (90, 255, 90), 4, cv2.LINE_AA)
        xy, hd = paths[pick][0], paths[pick][1]
        ex, ey = np.interp(marker_t, hd, xy[:, 0]), np.interp(marker_t, hd, xy[:, 1])
        d = np.array([np.interp(marker_t + 0.05, hd, xy[:, 0]) - ex, np.interp(marker_t + 0.05, hd, xy[:, 1]) - ey])
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0
        off = EGO_L / 2 - EGO_REAR
        car = box_corners(ex + off * math.cos(yaw), ey + off * math.sin(yaw), yaw, EGO_L, EGO_W)
        cv2.fillPoly(canvas, [bev.poly(car)], (90, 255, 90), cv2.LINE_AA)
        cv2.polylines(canvas, [bev.poly(car)], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [bev.poly(box_corners(off, 0.0, 0.0, EGO_L, EGO_W))], True, (230, 230, 235), 2, cv2.LINE_AA)
        text(canvas, f"scene {i + 1}/{n}", (fx + 14, fy + 62), 0.7, (230, 230, 235))
        text(canvas, f"t = {marker_t:3.1f} s", (fx + 14, fy + 92), 0.7, (230, 230, 235))
        ly = by + 650
        for j, (lab, col) in enumerate((("selected", (90, 255, 90)), ("best in hindsight", (255, 0, 255)), ("human future", (255, 255, 255)))):
            cv2.line(canvas, (bx + 12, ly + 18 * j - 4), (bx + 40, ly + 18 * j - 4), col, 3, cv2.LINE_AA)
            text(canvas, lab, (bx + 48, ly + 18 * j), 0.45, (220, 220, 225), 1, False)
        for j in range(6):
            cv2.rectangle(canvas, (bx + 330 + j * 18, ly - 12), (bx + 346 + j * 18, ly + 2), turbo(0.15 + 0.85 * j / 5), -1)
        text(canvas, "hypotheses: predicted PDMS low -> high", (bx + 330, ly + 20), 0.4, (220, 220, 225), 1, False)
        text(canvas, "NAV COMMAND (agent input)", (ix + 14, iy + 28), 0.55, (200, 200, 205), 1, False)
        for j, nm in enumerate(CMD[:3]):
            on = (cmd == j)
            x0 = ix + 14 + j * 150
            cv2.rectangle(canvas, (x0, iy + 40), (x0 + 140, iy + 78), (90, 255, 90) if on else (60, 60, 66), -1 if on else 1)
            text(canvas, nm, (x0 + 10, iy + 66), 0.6, (0, 0, 0) if on else (160, 160, 165), 2 if on else 1, False)
        if cmd == 3:
            text(canvas, "UNKNOWN", (ix + 14, iy + 100), 0.6, (90, 255, 90))
        text(canvas, f"ego speed {ego_v * 3.6:5.1f} km/h", (ix + 14, iy + 112), 0.6)
        text(canvas, f"{ld.log[:10]}..{ld.log[-11:]}", (ix + 14, iy + 136), 0.45, (150, 150, 155), 1, False)
        text(canvas, f"selected  PDMS {100 * pdms[pick]:5.1f}   (hypothesis #{pick})", (ix + 14, iy + 170), 0.6, (90, 255, 90))
        text(canvas, f"best      PDMS {100 * pdms[best]:5.1f}", (ix + 14, iy + 196), 0.6, (255, 120, 255))
        text(canvas, f"mean of 64    {100 * float(np.mean(pdms[valid])):5.1f}", (ix + 14, iy + 222), 0.6)
        for j, nm in enumerate(("NC", "DAC", "EP", "TTC", "C", "DDC")):
            v = float(sub[pick][j])
            x0 = ix + 14 + (j % 3) * 150
            y0 = iy + 250 + (j // 3) * 40
            cv2.rectangle(canvas, (x0, y0), (x0 + 130, y0 + 12), (60, 60, 66), 1)
            cv2.rectangle(canvas, (x0, y0), (x0 + int(130 * v), y0 + 12), (90, 255, 90) if v > 0.99 else (60, 90, 255), -1)
            text(canvas, f"{nm} {v:.2f}", (x0, y0 + 28), 0.45, (210, 210, 215), 1, False)
        run = (run_stats["sum_sel"] + pdms[pick]) / (i + 1)
        text(canvas, f"running PDMS (selected) {100 * run:5.1f}", (ix + 14, iy + 340), 0.55, (230, 230, 235))
        cv2.rectangle(canvas, (14, 1050), (1426, 1062), (60, 60, 66), 1)
        cv2.rectangle(canvas, (14, 1050), (14 + int(1412 * (i + p) / n), 1062), (90, 255, 90), -1)
        text(canvas, "REFe final model (10,075 steps)  |  NAVSIM navtest subset, 200 tokens  |  open-loop plan, harness-scored", (14, 1030), 0.55, (200, 200, 205), 1, False)
        writer.stdin.write(canvas.tobytes())
    run_stats["sum_sel"] += float(pdms[pick])


def intro_frames(writer, point, n):
    for f in range(FPS * 5):
        c = np.full((H, W, 3), 18, np.uint8)
        a = min(1.0, f / (FPS * 0.8))

        def sh(col):
            return tuple(int(v * a) for v in col)
        text(c, "TanitAD  -  REFe final model: what the planner sees and chooses", (90, 200), 1.4, sh((240, 240, 245)), 3, False)
        text(c, f"{n} NAVSIM navtest scenes (200-token subset, 93 logs); plans scored by NAVSIM's own PDM harness", (90, 270), 0.85, sh((200, 200, 205)), 2, False)
        text(c, f"Subset PDMS {point['pdms']:.2f}  [{point['lo']:.2f}, {point['hi']:.2f}]   (+{point['vs_stop']:.1f} vs standing still); the full 12,146-token result is reported separately",
             (90, 320), 0.8, sh((90, 255, 90)), 2, False)
        for j, (lab, col2) in enumerate((("selected plan: the hypothesis the scorer picked (green ribbon on the road)", (90, 255, 90)),
                                         ("64 trajectory hypotheses: colour = the scorer's predicted PDMS, blue low -> yellow/red high", (60, 160, 255)),
                                         ("best in hindsight: the hypothesis with the highest TRUE PDMS (magenta)", (255, 0, 255)),
                                         ("human future: what the logged driver did (white)", (255, 255, 255)),
                                         ("NAV COMMAND: NAVSIM's driving command is an agent input; the BEV shows logged agents (not a model input)", (200, 200, 205)))):
            ys = 420 + 46 * j
            cv2.line(c, (90, ys - 6), (150, ys - 6), sh(col2), 5, cv2.LINE_AA)
            text(c, lab, (175, ys), 0.8, sh((225, 225, 230)), 1, False)
        text(c, "Cameras: front (large), left / right / rear below. Banked model_final proposals and harness scores; nothing re-inferred for this video.",
             (90, 720), 0.65, sh((150, 150, 155)), 1, False)
        writer.stdin.write(c.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_navtest_sub200.mp4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--still", type=int, default=None, help="render only scene K and save one frame to --still-out")
    ap.add_argument("--still-out", default="still.png")
    a = ap.parse_args()
    P, T, exp = load_inputs()
    toks = [str(t) for t in P["token"]]
    n = len(toks) if a.limit is None else min(a.limit, len(toks))
    pt = json.load(open(f"{DATA}/points/sub200_final.json", encoding="utf-8"))
    x = pt["floors"]["arms"]["REFe"]["interval"]
    pair = pt["floors"]["pairs"]["REFe__minus__STOP"]
    point = {"pdms": pt["score"]["summary_x100_4dp"]["PDMS"], "lo": 100 * x["lo"], "hi": 100 * x["hi"], "vs_stop": pair["delta_x100"]}
    if a.still is not None:
        class Sink:
            class stdin:
                frames = []

                @staticmethod
                def write(b):
                    Sink.stdin.frames.append(b)
        i = a.still
        render_scene(LogData(exp[toks[i]]["log_name"]), toks[i], i, n, P, T, exp, {"sum_sel": 0.0}, Sink)
        fr = Sink.stdin.frames[int(len(Sink.stdin.frames) * 0.65)]
        cv2.imwrite(a.still_out, np.frombuffer(fr, np.uint8).reshape(H, W, 3))
        print("wrote", a.still_out)
        return 0
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    intro_frames(writer, point, n)
    stats = {"sum_sel": 0.0}
    cur, ld = None, None
    for i in range(n):
        log = exp[toks[i]]["log_name"]
        if log != cur:
            ld, cur = LogData(log), log
        render_scene(ld, toks[i], i, n, P, T, exp, stats, writer)
        if i % 10 == 0:
            print(f"scene {i + 1}/{n}", flush=True)
    writer.stdin.close()
    writer.wait()
    print("ZZVIDEO_DONE", a.out, f"scenes {n}", f"mean selected PDMS over rendered scenes {100 * stats['sum_sel'] / n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
