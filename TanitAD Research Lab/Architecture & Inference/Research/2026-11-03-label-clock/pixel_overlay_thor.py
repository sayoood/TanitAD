#!/usr/bin/env python
"""r9clk probe 4 (pixel truth) -- does the banked label sit ON the object in the SAME training image?

For a STATIC parked car seen at 8-17 m (2-9 m lateral) while the ego drives > 9 m/s, project the banked 3-D label (2-D join cx, cy, yaw, l, w +
3-D join cz, h; rig frame of raw frame i) into the v2ep image of raw frame i (the NEWEST frame of the stacked row) with the
clip's own camera extrinsics and the cylindrical canonical frame. Draw two cuboids:
  GREEN  the banked label as it is (delta 0)
  RED    what a label that is 0.2 s OLD would be: the same static world point expressed in the ego frame of T_i - 0.2 s
         (x grows by ~ 0.2 v), plotted in the frame-i rig.
If the labels lag the image, RED sits on the car and GREEN is ~0.2 v too near. Metric: pixel distance from the car's
image footprint is judged by eye (this is the one probe that is not a clock comparison) and, numerically, by the projected
centre's column/row offset between GREEN and RED.
Run on Thor (CPU): OMP_NUM_THREADS=1 python pixel_overlay_thor.py --out-dir /home/nvidia/r9clk/overlay
"""
import argparse
import io
import json
import lzma
import math
import os
import tarfile
import hashlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torchvision.io as tvio
from PIL import Image, ImageDraw

D = "/home/nvidia/data"
JOIN2D = f"{D}/joins/b1_train_plus_eval_agents.jsonl.xz"
JOIN3D = f"{D}/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
EGO, OBS = f"{D}/obstacle/egomotion_alpamayo", f"{D}/obstacle/obstacle_offline_b1train"
V2EP = f"{D}/refcv8-b1-416x1024-train-4572"
EXTR = f"{D}/refcv8_train4572_eval139_extrinsics.json"
TAR = f"{D}/b1-bundle/timestamps/timestamps.tar"


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def yaw_q(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Ego:
    def __init__(self, df):
        ts = df["timestamp"].to_numpy(np.float64); o = np.argsort(ts)
        self.t = ts[o] / 1e6
        self.x = df["x"].to_numpy(np.float64)[o]; self.y = df["y"].to_numpy(np.float64)[o]
        self.yaw = np.unwrap(yaw_q(*(df[c].to_numpy(np.float64)[o] for c in ("qx", "qy", "qz", "qw"))))

    def at(self, t):
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), float(np.interp(t, self.t, self.yaw))


def e2w(p, c, a):
    cs, sn = math.cos(p[2]), math.sin(p[2]); return p[0] + c * cs - a * sn, p[1] + c * sn + a * cs


def w2e(p, wx, wy):
    cs, sn = math.cos(p[2]), math.sin(p[2]); dx, dy = wx - p[0], wy - p[1]; return dx * cs + dy * sn, -dx * sn + dy * cs


def project(P_rig, R_c2r, t_c, f, W, H):
    pc = (P_rig - t_c) @ R_c2r                      # = R^T (p - t) for row vectors
    # camera frame: x right, y down, z forward (RIG_TO_CAM)
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    rho = np.hypot(x, z)
    col = (W - 1) / 2 + f * np.arctan2(x, z)
    row = (H - 1) / 2 + f * y / np.maximum(rho, 1e-6)
    return col, row, z > 0.5


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def cuboid(cx, cy, yaw, l, w, cz, h):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for zz in (cz - h / 2, cz + h / 2):
        for dx, dy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
            pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c, zz))
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=120)
    ap.add_argument("--n-examples", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    extr = json.load(open(EXTR))
    tf = tarfile.open(TAR); tn = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.name.endswith(".parquet")}
    # lockstep 2-D / 3-D join read, grouped by clip
    clips, order = {}, []
    with lzma.open(JOIN2D, "rt") as f2, lzma.open(JOIN3D, "rt") as f3:
        for l2, l3 in zip(f2, f3):
            r2 = json.loads(l2)
            c = r2["clip_id"]
            if c not in clips:
                if len(clips) >= a.n_clips:
                    break
                clips[c] = []
            r3 = json.loads(l3)
            clips[c].append((r2, r3))
    out, done = [], 0
    for cid, recs in clips.items():
        if done >= a.n_examples:
            break
        if cid not in extr or f"{cid}.timestamps.parquet" not in tn or not os.path.isfile(f"{V2EP}/{cid}.v2ep.pt"):
            continue
        ego = Ego(pd.read_parquet(f"{EGO}/{cid}.parquet"))
        obs = pd.read_parquet(f"{OBS}/{cid}.parquet")
        to = obs["timestamp_us"].to_numpy(np.float64) / 1e6
        ox = np.interp(to, ego.t, ego.x); oy = np.interp(to, ego.t, ego.y); oyaw = np.interp(to, ego.t, ego.yaw)
        cs, sn = np.cos(oyaw), np.sin(oyaw)
        cxr, cyr = obs["center_x"].to_numpy(np.float64), obs["center_y"].to_numpy(np.float64)
        wx, wy = ox + cxr * cs - cyr * sn, oy + cxr * sn + cyr * cs
        tid = obs["track_id"].astype(str).to_numpy()
        static = set()
        for k in np.unique(tid):
            m = tid == k
            if m.sum() >= 25 and to[m].max() - to[m].min() >= 2.5:
                rad = np.hypot(wx[m] - np.median(wx[m]), wy[m] - np.median(wy[m]))
                if np.quantile(rad, .95) <= 0.15:        # stricter than the scan: parked
                    static.add(k)
        tdf = pd.read_parquet(io.BytesIO(tf.extractfile(tn[f"{cid}.timestamps.parquet"]).read()))
        tc = next(c for c in tdf.columns if "time" in c.lower())
        tf_ = tdf[tc].to_numpy(np.float64); span = tf_[-1] - tf_[0]
        unit = next((u for u in (1e9, 1e6, 1e3) if span / u > 1.0), 1.0)
        n_t = max(int(span / unit * 10), 4)
        T = np.linspace(tf_[0], tf_[-1], n_t) / unit
        d = torch.load(f"{V2EP}/{cid}.v2ep.pt", map_location="cpu", weights_only=False, mmap=True)
        fr = d["frame"]; Hh, Ww, f_ref = int(fr["height"]), int(fr["width"]), float(fr["f_ref"])
        e = extr[cid]; R = quat_to_R(e["qx"], e["qy"], e["qz"], e["qw"]); tc_ = np.array([e["x"], e["y"], e["z"]])
        pick = None
        for r2, r3 in recs[40:160:3]:
            i = r2["frame"]
            if i >= n_t:
                continue
            v = float(d["poses"][i, 3])
            if v < 9.0:
                continue
            for a2, a3 in zip(r2["agents"], r3["agents"]):
                if a2["track_id"] in static and a2["cls"] == "automobile" and 8 < a2["cx"] < 17 and 2.0 < abs(a2["cy"]) < 9:
                    col, row, ok = project(cuboid(a2["cx"], a2["cy"], a2["yaw"], a2["l"], a2["w"], a3["cz"], a3["h"])[:, :], R, tc_, f_ref, Ww, Hh)
                    if ok.all() and 0 < col.min() and col.max() < Ww - 1 and 0 < row.min() and row.max() < Hh - 1:
                        pick = (i, a2, a3, v); break
            if pick:
                break
        if not pick:
            continue
        i, a2, a3, v = pick
        # image of raw frame i
        jl = d["jpeg_len"].to(torch.int64); offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(jl, 0)])
        buf = d["jpeg_buf"][offs[i]:offs[i + 1]]
        img = (tvio.decode_png(buf) if d.get("codec", "png") == "png" else tvio.decode_jpeg(buf)).permute(1, 2, 0).numpy()
        im = Image.fromarray(img).convert("RGB"); dr = ImageDraw.Draw(im)
        # RED: label as an 0.2 s OLD label would be
        pose_i = ego.at(T[i]); wx_, wy_ = e2w(pose_i, a2["cx"], a2["cy"])
        pose_old = ego.at(T[i] - 0.2); ox_, oy_ = w2e(pose_old, wx_, wy_)
        yaw_old = a2["yaw"] + (pose_i[2] - pose_old[2])
        res = {"clip": sha12(cid), "frame": int(i), "ego_speed_mps": v, "label_cx_cy": [a2["cx"], a2["cy"]],
               "old_label_cx_cy": [ox_, oy_], "old_minus_banked_x_m": ox_ - a2["cx"], "expected_0p2_v": 0.2 * v}
        for name, (cx, cy, yw), col_rgb in (("banked", (a2["cx"], a2["cy"], a2["yaw"]), (0, 255, 0)),
                                             ("old0p2s", (ox_, oy_, yaw_old), (255, 0, 0))):
            P = cuboid(cx, cy, yw, a2["l"], a2["w"], a3["cz"], a3["h"])
            col, row, ok = project(P, R, tc_, f_ref, Ww, Hh)
            for p, q in EDGES:
                dr.line([(col[p], row[p]), (col[q], row[q])], fill=col_rgb, width=2)
            cc, rr, _ = project(np.array([[cx, cy, a3["cz"]]]), R, tc_, f_ref, Ww, Hh)
            res[f"{name}_centre_col_row"] = [float(cc[0]), float(rr[0])]
        c0, r0 = res["banked_centre_col_row"]
        x0, x1 = int(max(c0 - 260, 0)), int(min(c0 + 260, Ww)); y0, y1 = int(max(r0 - 150, 0)), int(min(r0 + 150, Hh))
        crop = im.crop((x0, y0, x1, y1))
        raw = Image.fromarray(img).convert("RGB").crop((x0, y0, x1, y1))
        panel = Image.new("RGB", (crop.width * 2, crop.height)); panel.paste(raw, (0, 0)); panel.paste(crop, (crop.width, 0))
        fn = f"{a.out_dir}/overlay_{sha12(cid)}_f{i}.jpg"; panel.save(fn, quality=88)
        res["file"] = os.path.basename(fn)
        out.append(res); done += 1
        print(json.dumps(res), flush=True)
    json.dump(out, open(f"{a.out_dir}/overlay_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
