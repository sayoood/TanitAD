#!/usr/bin/env python3
"""ov_audit.py -- INDEPENDENT audit of the refcv6 3-D box camera overlay, on a clip that has LiDAR (Thor).

Answers four questions with numbers, each against a reference that does NOT share the renderer's code:

  P1  pixel agreement: the renderer's own corner projection (render_refcv6_map_video.cuboid_corners +
      tanitad RigCamera built from the renderer's extrinsics table) vs an INDEPENDENT numpy projection written
      here from the RAW calibration parquet row (quaternion -> R by hand, cylindrical model by hand).
  P2  frame geometry against PIXELS: the cached 416x1024 frame vs this script's own re-rendering of the RAW
      front-wide mp4 through the clip's own f-theta intrinsics (camera_intrinsics parquet). If the canonical
      cylinder were pitched, rolled, off-centre or at a different f_ref, the minimum-error shift would not be
      (0, 0) and the error at (0, 0) would not be at the codec floor.
  P3  extrinsics against an INDEPENDENT SENSOR: LiDAR points (lidar_top_360fov, through ITS OWN extrinsic)
      drawn into the cached frame through the camera extrinsic; the GT cuboids beside them (PNG panels).
  P4  GT geometry: cuboid bottom (join3d cz - h/2) vs the LiDAR ground around the box; LiDAR points INSIDE
      each GT box vs the same box MIRRORED (y -> -y), by range band; join3d vs the raw obstacle.offline parquet.
  P5  cross-check with the SAM3 pipeline's own camera model (sam3map/camera_model.py, native f-theta) on the
      same 3-D points.

Prints / writes sha12 only; never a raw clip id.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import io
import json
import lzma
import math
import os
import sys
import time

import numpy as np


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------- #
# INDEPENDENT geometry: numpy only, written from the definitions, no tanitad import               #
# ---------------------------------------------------------------------------------------------- #
def R_from_quat(qx, qy, qz, qw):
    """Hamilton unit quaternion (x, y, z, w) -> the ACTIVE rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def cyl_project(P_rig, R_c2r, t, W, H, f):
    """rig points -> canonical cylindrical (u=col, v=row). Camera axes x right, y down, z optical."""
    Pc = (np.asarray(P_rig, np.float64) - t) @ R_c2r          # = R^T (p - t), row-wise
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    rho = np.hypot(x, z)
    u = (W - 1) / 2.0 + f * np.arctan2(x, z)
    v = (H - 1) / 2.0 + f * y / np.maximum(rho, 1e-9)
    ok = (rho > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return u, v, ok, Pc


def ftheta_from_cam(Pc, poly, cx, cy):
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    r = np.zeros_like(th)
    for c in reversed(poly):
        r = r * th + c
    k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
    return cx + x * k, cy + y * k


def cyl_pixel_ray(u, v, W, H, f):
    phi = (np.asarray(u, np.float64) - (W - 1) / 2.0) / f
    yn = (np.asarray(v, np.float64) - (H - 1) / 2.0) / f
    return np.stack([np.sin(phi), yn, np.cos(phi)], axis=-1)


def bilinear(img, u, v):
    """img [H, W, 3] float; (u, v) pixel-centre coordinates; returns [..., 3] and an inside mask."""
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du, dv = u - u0, v - v0
    inside = (u0 >= 0) & (v0 >= 0) & (u0 + 1 < W) & (v0 + 1 < H)
    u0c, v0c = np.clip(u0, 0, W - 2), np.clip(v0, 0, H - 2)
    a = img[v0c, u0c]
    b = img[v0c, u0c + 1]
    c = img[v0c + 1, u0c]
    d = img[v0c + 1, u0c + 1]
    out = (a * ((1 - du) * (1 - dv))[..., None] + b * (du * (1 - dv))[..., None]
           + c * ((1 - du) * dv)[..., None] + d * (du * dv)[..., None])
    return out, inside


def box_local(P, cx, cy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    return c * dx + s * dy, -s * dx + c * dy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--c8", required=True, help="8-hex file prefix of the LiDAR clip ON THOR (runtime only; artifacts carry sha12)")
    ap.add_argument("--tree", default="/home/nvidia/bha_2325/tree")
    ap.add_argument("--render-tool", default="/home/nvidia/bha_2325/ov/render_refcv6_map_video.py")
    ap.add_argument("--out-dir", default="/home/nvidia/bha_2325/ov/out")
    ap.add_argument("--n-overlay", type=int, default=6)
    ap.add_argument("--max-dt-s", type=float, default=0.055)
    ap.add_argument("--mp4-frames", type=int, default=6)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    t_start = time.time()
    for p in (os.path.join(a.tree, "stack"), os.path.join(a.tree, "stack", "scripts"),
              os.path.join(a.tree, "taniteval")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    assert os.path.abspath(tanitad.__file__).startswith(os.path.abspath(os.path.join(a.tree, "stack"))), \
        tanitad.__file__
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera
    from tanitad.data import agent_cuboid_gt as ACG
    from tanitad.data.v2_dataset import stable_episode_id
    import pyarrow.parquet as pq
    import pandas as pd
    from PIL import Image, ImageDraw
    spec = importlib.util.spec_from_file_location("rv_tool", a.render_tool)
    RV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RV)

    D, S = "/home/nvidia/data", "/home/nvidia/sam3map/data"
    lid = sorted(glob.glob(f"{S}/{a.c8}-*.lidar_top_360fov.parquet"))[0]
    cid = os.path.basename(lid).split(".")[0]
    s12 = sha12(cid)
    rec = {"clip_sha12": s12, "tool": "ov_audit.py", "tanitad_file": tanitad.__file__,
           "render_tool": a.render_tool, "render_tool_md5": hashlib.md5(open(a.render_tool, "rb").read()).hexdigest()}
    print(f"[ov] clip {s12}", flush=True)

    # ---- calibration: RAW parquet rows and the renderer's table entry --------------------------- #
    ex = pd.read_parquet(f"{S}/sensor_extrinsics.chunk_0768.parquet").reset_index()
    ex = ex[ex["clip_id"] == cid].set_index("sensor_name")
    cam_row = ex.loc["camera_front_wide_120fov"]
    lid_row = ex.loc["lidar_top_360fov"]
    it = pd.read_parquet(f"{S}/calib/camera_intrinsics.chunk_0768.parquet").reset_index()
    it = it[(it["clip_id"] == cid) & (it["camera_name"] == "camera_front_wide_120fov")].iloc[0]
    poly = tuple(float(it[f"fw_poly_{i}"]) for i in range(5))
    icx, icy, iw, ih = float(it["cx"]), float(it["cy"]), int(it["width"]), int(it["height"])
    table = json.load(open(f"{D}/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    te = table[cid]
    tab_vs_raw = {k: abs(float(te[k]) - float(cam_row[k])) for k in ("qx", "qy", "qz", "qw", "x", "y", "z")}
    rec["extrinsics_table_vs_raw_parquet_max_abs"] = max(tab_vs_raw.values())
    Rc = R_from_quat(float(cam_row.qx), float(cam_row.qy), float(cam_row.qz), float(cam_row.qw))
    tc = np.array([float(cam_row.x), float(cam_row.y), float(cam_row.z)])
    Rl = R_from_quat(float(lid_row.qx), float(lid_row.qy), float(lid_row.qz), float(lid_row.qw))
    tl = np.array([float(lid_row.x), float(lid_row.y), float(lid_row.z)])
    rec["camera"] = {"t_rig_m": tc.round(4).tolist(), "boresight_in_rig": (Rc @ [0, 0, 1]).round(5).tolist(),
                     "right_in_rig": (Rc @ [1, 0, 0]).round(5).tolist(), "down_in_rig": (Rc @ [0, 1, 0]).round(5).tolist(),
                     "pitch_down_deg": math.degrees(math.atan2(-(Rc @ [0, 0, 1])[2], math.hypot(*(Rc @ [0, 0, 1])[:2]))),
                     "intrinsics": {"cx": icx, "cy": icy, "w": iw, "h": ih, "fw_poly": poly}}
    rec["lidar"] = {"t_rig_m": tl.round(4).tolist(), "R_minus_I_max": float(np.abs(Rl - np.eye(3)).max())}

    # ---- the cached frames ------------------------------------------------------------------- #
    pay_p = sorted(glob.glob(f"{D}/refcv6-b1-416x1024-*/{cid}.v2ep.pt"))[0]
    pay = torch.load(pay_p, map_location="cpu", weights_only=False, mmap=True)
    fr = dict(pay["frame"])
    W, H, F = int(fr["width"]), int(fr["height"]), float(fr["f_ref"])
    k_stack = int(pay["n_stack"]) - 1
    offs = np.concatenate([[0], np.cumsum(pay["jpeg_len"].numpy())])
    buf = pay["jpeg_buf"]
    poses = pay["poses"].numpy()

    def raw_img(i):
        b = bytes(buf[int(offs[i]):int(offs[i + 1])].numpy().tobytes())
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    rec["cache"] = {"which": os.path.basename(os.path.dirname(pay_p)), "frame": fr, "n_raw": int(len(offs) - 1),
                    "n_stack": int(pay["n_stack"]), "codec": str(pay.get("codec"))}
    cam_frame = CanonicalFrame(height=H, width=W, f_ref=F, projection=str(fr["projection"]))
    rcam = RigCamera.from_extrinsics(FrontWideExtrinsics(
        qx=float(te["qx"]), qy=float(te["qy"]), qz=float(te["qz"]), qw=float(te["qw"]),
        x=float(te["x"]), y=float(te["y"]), z=float(te["z"])), cam_frame)

    # ---- the 2-D join (as written) and the 3-D join (the trainer's zh_for_frame) ------------- #
    t0 = time.time()
    recs = {}
    with lzma.open(f"{D}/joins/b1_train_plus_eval_agents.jsonl.xz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            if cid in ln:
                r = json.loads(ln)
                if r["clip_id"] == cid:
                    recs[int(r["frame_idx"])] = r
    print(f"[ov] join records {len(recs)} ({time.time() - t0:.0f} s)", flush=True)
    j3 = ACG.open_join3d(f"{D}/join3d/b1_train_plus_eval_agents_3d.jsonl.xz", clips={cid})
    rec["join"] = {"n_records": len(recs), "t_s_range": [min(r["t_s"] for r in recs.values()),
                                                         max(r["t_s"] for r in recs.values())]}

    # ---- raw obstacle.offline (for the join3d-vs-raw check) ------------------------------------ #
    obs_p = f"{D}/obstacle/obstacle_offline_b1train/{cid}.parquet"
    obs = pd.read_parquet(obs_p) if os.path.exists(obs_p) else None
    rec["raw_obstacle_columns"] = None if obs is None else list(obs.columns)

    # ---- LiDAR: spin clock + the pre-decoded sweeps (rig frame with z - 0.325, mapq_ours.Z) --------- #
    sp = pq.read_table(lid, columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
    mid_s = (np.asarray(sp["spin_start_timestamp"], np.int64) + np.asarray(sp["spin_end_timestamp"], np.int64)) / 2e6
    sweeps = {int(os.path.basename(p)[:4]): p for p in glob.glob(f"/home/nvidia/sam3map/data/sweeps/{a.c8}/*.npy")}
    rec["lidar"]["spin_mid_range_s"] = [float(mid_s[0]), float(mid_s[-1])]
    rec["lidar"]["n_predecoded_sweeps"] = len(sweeps)
    Z_SHIFT = 0.325

    # ---- per frame ----------------------------------------------------------------------------- #
    duv = []                       # renderer vs independent, per corner
    boxes = []                     # per GT agent row
    frames_used = []
    overlay_frames = []
    for f, r in sorted(recs.items()):
        si = int(np.abs(mid_s - float(r["t_s"])).argmin())
        dt = float(mid_s[si] - float(r["t_s"]))
        if abs(dt) > a.max_dt_s or si not in sweeps:
            continue
        pts = np.load(sweeps[si])[:, :3].astype(np.float64)
        pts[:, 2] += Z_SHIFT                                          # back to the rig frame
        ag = r["agents"]
        if not ag:
            continue
        tids = [str(d.get("track_id", "")) for d in ag]
        cz, hh, zm = ACG.zh_for_frame(cid, int(f) + k_stack, tids, join3d=j3)
        v_ego = float(poses[f + k_stack][3]) if poses.shape[1] > 3 else float("nan")
        frames_used.append({"f": int(f), "sweep": si, "dt_s": round(dt, 4), "n_agents": len(ag), "v_ms": v_ego})
        overlay_frames.append((f, si))
        for i, d in enumerate(ag):
            cx_, cy_, yaw, l, w = float(d["cx"]), float(d["cy"]), float(d["yaw"]), float(d["l"]), float(d["w"])
            rng = math.hypot(cx_, cy_)
            vis = (0 <= cx_ <= 60) and abs(cy_) <= 16 and math.atan2(abs(cy_), cx_) <= math.radians(60)
            row = {"f": int(f), "cls": str(d.get("cls", "")), "range": rng, "target": bool(vis),
                   "occ_flag": int(d.get("occ", -1)), "has_zh": bool(zm[i]), "l": l, "w": w, "v_ms": v_ego}
            if zm[i]:
                czi, hi = float(cz[i]), float(hh[i])
                row.update(cz=czi, h=hi, bottom=czi - hi / 2)
                corners = RV.cuboid_corners(cx_, cy_, czi, l, w, hi, yaw)
                col, rowp, ok = rcam.project(torch.as_tensor(corners, dtype=torch.float64))
                u2, v2, ok2, _ = cyl_project(corners, Rc, tc, W, H, F)
                both = ok.numpy().astype(bool) & ok2
                if both.any():
                    duv.append(np.c_[np.abs(col.numpy()[both] - u2[both]), np.abs(rowp.numpy()[both] - v2[both])])
                row["n_corners_in_frame"] = int(both.sum())
            # ground ring + inside counts, in the box's own frame
            xl, yl = box_local(pts, cx_, cy_, yaw)
            ring = ((np.abs(xl) <= l / 2 + 3.0) & (np.abs(yl) <= w / 2 + 3.0)
                    & ~((np.abs(xl) <= l / 2 + 1.0) & (np.abs(yl) <= w / 2 + 1.0)))
            zr = pts[ring, 2]
            row["n_ring"] = int(ring.sum())
            if zr.size >= 20:
                row["ground_p10"] = float(np.percentile(zr, 10))
                row["ground_p25"] = float(np.percentile(zr, 25))
            g = row.get("ground_p10", 0.0)
            top = row.get("cz", 0.8) + row.get("h", 1.6) / 2
            inside = (np.abs(xl) <= l / 2 + 0.3) & (np.abs(yl) <= w / 2 + 0.3) & (pts[:, 2] >= g + 0.25) & (pts[:, 2] <= top + 0.3)
            row["n_in"] = int(inside.sum())
            xm, ym = box_local(pts, cx_, -cy_, -yaw)                     # the MIRROR control
            inside_m = (np.abs(xm) <= l / 2 + 0.3) & (np.abs(ym) <= w / 2 + 0.3) & (pts[:, 2] >= g + 0.25) & (pts[:, 2] <= top + 0.3)
            row["n_in_mirror"] = int(inside_m.sum())
            # does the mirrored footprint overlap a real GT footprint in this frame? (then the control is spoiled)
            row["mirror_hits_gt"] = any(math.hypot(float(e["cx"]) - cx_, float(e["cy"]) + cy_) < 0.5 * (l + float(e["l"]))
                                        for e in ag)
            boxes.append(row)
    rec["frames_used"] = len(frames_used)
    rec["frames_dt_s_abs_max"] = max(abs(x["dt_s"]) for x in frames_used) if frames_used else None
    print(f"[ov] frames with a sweep: {len(frames_used)}; GT rows {len(boxes)} ({time.time() - t_start:.0f} s)",
          flush=True)

    # ---- P1 ----------------------------------------------------------------------------------- #
    if duv:
        dd = np.concatenate(duv)
        rec["P1_pixels_renderer_vs_independent"] = {
            "n_corners": int(dd.shape[0]), "n_boxes": int(sum(1 for b in boxes if b.get("n_corners_in_frame"))),
            "max_abs_du_px": float(dd[:, 0].max()), "max_abs_dv_px": float(dd[:, 1].max()),
            "p99_abs_du_px": float(np.percentile(dd[:, 0], 99)), "p99_abs_dv_px": float(np.percentile(dd[:, 1], 99))}

    # ---- P4: z and object evidence --------------------------------------------------------------- #
    def band(x):
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 1e9)):
            if lo <= x < hi:
                return f"{lo}-{hi if hi < 1e9 else 'inf'}m"
    zb = [b for b in boxes if b.get("has_zh") and "ground_p10" in b]
    ev = {}
    for b in boxes:
        k = (band(b["range"]), "target" if b["target"] else "filtered")
        e = ev.setdefault(k, {"n": 0, "evid3": 0, "evid10": 0, "mirror3": 0, "mirror_n": 0})
        e["n"] += 1
        e["evid3"] += int(b["n_in"] >= 3)
        e["evid10"] += int(b["n_in"] >= 10)
        if not b["mirror_hits_gt"]:
            e["mirror_n"] += 1
            e["mirror3"] += int(b["n_in_mirror"] >= 3)
    rec["P4_object_evidence"] = {f"{k[0]}|{k[1]}": {**v, "frac_evid3": v["evid3"] / v["n"] if v["n"] else None,
                                                    "frac_evid10": v["evid10"] / v["n"] if v["n"] else None,
                                                    "frac_mirror3": v["mirror3"] / v["mirror_n"] if v["mirror_n"] else None}
                                 for k, v in sorted(ev.items())}
    bycls = {}
    for b in zb:
        bycls.setdefault(b["cls"], []).append(b["bottom"] - b["ground_p10"])
    rec["P4_bottom_minus_lidar_ground_p10"] = {c: {"n": len(v), "median": float(np.median(v)),
                                                   "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
                                               for c, v in sorted(bycls.items())}
    near = [b for b in zb if b["range"] < 20]
    rec["P4_ground_p10_abs_rig_z_near20m"] = {"n": len(near), "median": float(np.median([b["ground_p10"] for b in near])) if near else None}
    rec["P4_bottom_abs_rig_z_by_cls"] = {c: float(np.median([b["bottom"] for b in zb if b["cls"] == c]))
                                         for c in sorted(set(b["cls"] for b in zb))}
    # join3d vs raw obstacle.offline, same track, nearest sample
    if obs is not None:
        diffs = []
        tcol = obs["timestamp_us"].to_numpy() / 1e6
        for f, r in list(sorted(recs.items()))[::10]:
            tids = [str(d.get("track_id", "")) for d in r["agents"]]
            cz, hh, zm = ACG.zh_for_frame(cid, int(f) + k_stack, tids, join3d=j3)
            for i, tid in enumerate(tids):
                if not zm[i]:
                    continue
                sel = (obs["track_id"].astype(str).to_numpy() == tid)
                if not sel.any():
                    continue
                j = np.nonzero(sel)[0][np.abs(tcol[sel] - float(r["t_s"])).argmin()]
                diffs.append((abs(float(obs["center_z"].iloc[j]) - float(cz[i])),
                              abs(float(obs["size_z"].iloc[j]) - float(hh[i]))))
        if diffs:
            dd = np.asarray(diffs)
            rec["P4_join3d_vs_raw_parquet"] = {"n": int(dd.shape[0]), "max_abs_dcz_m": float(dd[:, 0].max()),
                                               "max_abs_dh_m": float(dd[:, 1].max()),
                                               "median_abs_dcz_m": float(np.median(dd[:, 0]))}

    # ---- P2: the cached frame vs this script's own re-render of the RAW mp4 ----------------------- #
    try:
        import av
        mp4 = sorted(glob.glob(f"{S}/frontwide/{cid}.camera_front_wide_120fov.mp4"))
        tsp = sorted(glob.glob(f"{S}/frontwide/{cid}.camera_front_wide_120fov.timestamps.parquet"))
        if mp4 and tsp:
            ts = pd.read_parquet(tsp[0])
            tcol = [c for c in ts.columns if "timestamp" in c][0]
            t_cam = ts[tcol].to_numpy().astype(np.int64)
            n_raw = int(len(offs) - 1)
            span = (t_cam[-1] - t_cam[0]) / 1e6
            t_q = np.linspace(t_cam[0], t_cam[-1], int(span * 10))
            rec["P2_timing"] = {"n_cam_frames": int(t_cam.size), "n_query": int(t_q.size), "n_raw_cached": n_raw}
            want = sorted(set(int(x) for x in np.linspace(10, n_raw - 10, a.mp4_frames)))
            cont = av.open(mp4[0])
            frames_np = {}
            idx_needed = {}
            for k in want:
                j = int(np.searchsorted(t_cam, t_q[k])) if k < t_q.size else None
                if j is None:
                    continue
                for jj in (j - 1, j, j + 1):
                    idx_needed.setdefault(jj, []).append(k)
            for n_f, frm in enumerate(cont.decode(video=0)):
                if n_f in idx_needed:
                    frames_np[n_f] = frm.to_ndarray(format="rgb24").astype(np.float32)
                if n_f > max(idx_needed):
                    break
            dec_h, dec_w = next(iter(frames_np.values())).shape[:2]
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            p2 = []
            for k in want:
                cached = raw_img(k).astype(np.float32)
                best = None
                for jj in sorted(j_ for j_, ks in idx_needed.items() if k in ks):
                    if jj not in frames_np:
                        continue
                    shifts = {}
                    for sdu in (-3, 0, 3):
                        for sdv in (-3, 0, 3):
                            ray = cyl_pixel_ray(uu + sdu, vv + sdv, W, H, F).reshape(-1, 3)
                            un, vn = ftheta_from_cam(ray, poly, icx, icy)
                            un, vn = un * dec_w / iw, vn * dec_h / ih
                            smp, ins = bilinear(frames_np[jj], un, vn)
                            obsm = ins & (ray[:, 2] > 0)
                            mae = float(np.abs(smp[obsm] - cached.reshape(-1, 3)[obsm]).mean())
                            shifts[(sdu, sdv)] = mae
                    k0 = min(shifts, key=shifts.get)
                    item = {"k": k, "mp4_idx_offset": jj - int(np.searchsorted(t_cam, t_q[k])),
                            "mae_at_0_0": shifts[(0, 0)], "best_shift": list(k0), "mae_best": shifts[k0],
                            "mae_all": {f"{s_[0]},{s_[1]}": round(v_, 3) for s_, v_ in shifts.items()}}
                    if best is None or item["mae_at_0_0"] < best["mae_at_0_0"]:
                        best = item
                if best:
                    p2.append(best)
            rec["P2_cache_vs_own_rerender_of_raw_mp4"] = {"decoded_hw": [dec_h, dec_w], "per_frame": p2}
    except Exception as e:                                           # noqa: BLE001
        rec["P2_error"] = f"{type(e).__name__}: {e}"

    # ---- P5: the SAM3 pipeline's own camera model on the same 3-D points --------------------------- #
    try:
        sys.path.insert(0, "/home/nvidia/sam3map")
        import camera_model as CM
        cams = sorted(glob.glob(f"/home/nvidia/sam3map/native7/seq_{a.c8}/*/calib.npz"))
        c = np.load(cams[0], allow_pickle=True)
        names = [str(x) for x in c["cam_names"]] if "cam_names" in c.files else None
        rec["P5_calib_npz_keys"] = list(c.files)
        if names and "camera_front_wide_120fov" in names:
            idx = names.index("camera_front_wide_120fov")
            rec["P5_cam_index_rule"] = "cam_names"
        else:                        # identify the front-wide by its OWN principal point (per-clip, unique here)
            idx = int(np.argmin(np.abs(c["ftheta_cx"] - icx) + np.abs(c["ftheta_cy"] - icy)))
            rec["P5_cam_index_rule"] = (f"argmin |ftheta_cx - cx| + |ftheta_cy - cy| -> {idx} "
                                        f"(residual {float(abs(c['ftheta_cx'][idx] - icx) + abs(c['ftheta_cy'][idx] - icy)):.4f} px)")
        scam = CM.Camera.from_calib(c, idx)
        R_cl, t_cl = Rl.T @ Rc, Rl.T @ (tc - tl)                     # the camera pose in the LIDAR frame
        rec["P5_sam3_calib_vs_parquet"] = {"R_max_abs_vs_cam2rig": float(np.abs(scam.R - Rc).max()),
                                           "t_max_abs_vs_cam_in_rig": float(np.abs(scam.t - tc).max()),
                                           "R_max_abs_vs_cam2lidar": float(np.abs(scam.R - R_cl).max()),
                                           "t_max_abs_vs_cam_in_lidar": float(np.abs(scam.t - t_cl).max()),
                                           "sam3_t": scam.t.round(4).tolist(),
                                           "ft": {k: (list(v) if isinstance(v, tuple) else v) for k, v in scam.ft.items()}}
        rng_ = np.random.default_rng(0)
        P = np.c_[rng_.uniform(3, 60, 5000), rng_.uniform(-20, 20, 5000), rng_.uniform(-0.5, 3.0, 5000)]
        u2, v2, ok2, Pc = cyl_project(P, Rc, tc, W, H, F)
        ray = cyl_pixel_ray(u2, v2, W, H, F)
        un, vn = ftheta_from_cam(ray, poly, icx, icy)                # canonical pixel -> native pixel
        # the SAM3 calib is named `sensor2lidar_*`: try the points in the RIG frame and in the LIDAR frame
        # (P_lidar = Rl^T (P_rig - tl)) and report both, so a frame mismatch cannot hide in one number
        for tag, PP in (("points_in_rig_frame", P), ("points_in_lidar_frame", (P - tl) @ Rl)):
            us, vs, oks = scam.project_rig(PP)
            both = oks & ok2
            rec.setdefault("P5_native_px_sam3_vs_ours", {})[tag] = {
                "n_points": int(both.sum()),
                "max_abs_du_native_px": float(np.abs(us[both] - un[both]).max()) if both.any() else None,
                "max_abs_dv_native_px": float(np.abs(vs[both] - vn[both]).max()) if both.any() else None}
    except Exception as e:                                           # noqa: BLE001
        rec["P5_error"] = f"{type(e).__name__}: {e}"

    # ---- P3: overlay panels (LiDAR through its own extrinsic + GT cuboids, renderer drawing) ---------- #
    sel = overlay_frames[:: max(1, len(overlay_frames) // max(1, a.n_overlay))][: a.n_overlay]
    for f, si in sel:
        img = Image.fromarray(raw_img(f + k_stack)).convert("RGB")
        dr = ImageDraw.Draw(img, "RGBA")
        pts = np.load(sweeps[si])[:, :3].astype(np.float64)
        pts[:, 2] += Z_SHIFT
        u2, v2, ok2, Pc = cyl_project(pts, Rc, tc, W, H, F)
        dist = np.linalg.norm(Pc, axis=1)
        m = ok2 & (Pc[:, 2] > 0.5)
        order = np.argsort(-dist[m])
        uu_, vv_, dd_, zz_ = u2[m][order], v2[m][order], dist[m][order], pts[m][order][:, 2]
        for x_, y_, d_, z_ in zip(uu_, vv_, dd_, zz_):
            if z_ < 0.25:
                col_ = (90, 90, 255, 70)                              # ground returns: faint blue
            else:
                t_ = min(1.0, d_ / 40.0)
                col_ = (int(255 * (1 - t_)), int(255 * t_), 40, 230)  # obstacle returns: red (near) -> green (far)
            dr.point((float(x_), float(y_)), fill=col_)
        for d in recs[f]["agents"]:
            tids = [str(d.get("track_id", ""))]
            cz, hh, zm = ACG.zh_for_frame(cid, int(f) + k_stack, tids, join3d=j3)
            if not zm[0]:
                continue
            cx_, cy_ = float(d["cx"]), float(d["cy"])
            vis = (0 <= cx_ <= 60) and abs(cy_) <= 16 and math.atan2(abs(cy_), cx_) <= math.radians(60)
            RV.draw_cuboid_cam(dr, rcam, RV.cuboid_corners(cx_, cy_, float(cz[0]), float(d["l"]), float(d["w"]),
                                                           float(hh[0]), float(d["yaw"])),
                               (255, 255, 255, 255) if vis else (255, 170, 0, 200), width=1)
        img = img.resize((W * 3 // 2, H * 3 // 2), Image.BILINEAR)
        img.save(os.path.join(a.out_dir, f"ov_{s12}_f{f:03d}.png"))
    rec["overlay_pngs"] = [f"ov_{s12}_f{f:03d}.png" for f, _ in sel]
    rec["frames"] = frames_used
    json.dump(boxes, open(os.path.join(a.out_dir, f"ov_boxes_{s12}.json"), "w"), default=float)
    rec["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rec, open(os.path.join(a.out_dir, f"ov_audit_{s12}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("frames",)}, indent=1, default=float)[:9000], flush=True)


if __name__ == "__main__":
    main()
