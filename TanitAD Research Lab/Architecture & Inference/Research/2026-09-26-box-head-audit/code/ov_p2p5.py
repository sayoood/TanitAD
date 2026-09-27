#!/usr/bin/env python3
"""ov_p2p5.py -- overlay audit parts P2 and P5 (split out of ov_audit.py: P2 needs torch + PyAV (tanitad-train),
P5 needs scipy for the SAM3 pipeline's own camera model (tanitad-edge); the two venvs are never mixed).

P2  the cached 416x1024 frame vs THIS script's own re-render of the RAW front-wide mp4 through the clip's own
    f-theta intrinsics (camera_intrinsics parquet): mean |RGB| error at the declared geometry and at shifts of the
    canonical grid; the minimum must sit at (0, 0) if the cylinder's centre, f_ref and orientation are as declared.
P5  the SAM3 pipeline's camera model (sam3map/camera_model.py, native f-theta) vs ours (canonical cylinder ->
    ray -> the same f-theta), on the same 3-D points, with the SAM3 pipeline's OWN calib.npz for this clip.
sha12 only in the output.
"""
import argparse
import glob
import hashlib
import io
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ov_audit import R_from_quat, cyl_project, cyl_pixel_ray, ftheta_from_cam, bilinear  # noqa: E402


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def calib(cid, S):
    import pandas as pd
    ex = pd.read_parquet(f"{S}/sensor_extrinsics.chunk_0768.parquet").reset_index()
    ex = ex[ex["clip_id"] == cid].set_index("sensor_name")
    c, l_ = ex.loc["camera_front_wide_120fov"], ex.loc["lidar_top_360fov"]
    it = pd.read_parquet(f"{S}/calib/camera_intrinsics.chunk_0768.parquet").reset_index()
    it = it[(it["clip_id"] == cid) & (it["camera_name"] == "camera_front_wide_120fov")].iloc[0]
    return (R_from_quat(c.qx, c.qy, c.qz, c.qw), np.array([c.x, c.y, c.z], float),
            R_from_quat(l_.qx, l_.qy, l_.qz, l_.qw), np.array([l_.x, l_.y, l_.z], float),
            tuple(float(it[f"fw_poly_{i}"]) for i in range(5)), float(it.cx), float(it.cy), int(it.width), int(it.height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("p2", "p5"), required=True)
    ap.add_argument("--c8", required=True, help="8-hex file prefix of the LiDAR clip ON THOR (runtime only; artifacts carry sha12)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=6)
    a = ap.parse_args()
    S, D = "/home/nvidia/sam3map/data", "/home/nvidia/data"
    lid = sorted(glob.glob(f"{S}/{a.c8}-*.lidar_top_360fov.parquet"))[0]
    cid = os.path.basename(lid).split(".")[0]
    Rc, tc, Rl, tl, poly, icx, icy, iw, ih = calib(cid, S)
    W, H, F = 1024, 416, 488.92398517830253
    rec = {"clip_sha12": sha12(cid), "part": a.part}
    if a.part == "p2":
        import av
        import pandas as pd
        import torch
        from PIL import Image
        pay = torch.load(sorted(glob.glob(f"{D}/refcv6-b1-416x1024-*/{cid}.v2ep.pt"))[0], map_location="cpu",
                         weights_only=False, mmap=True)
        fr = dict(pay["frame"])
        assert (int(fr["width"]), int(fr["height"])) == (W, H) and abs(float(fr["f_ref"]) - F) < 1e-9, fr
        offs = np.concatenate([[0], np.cumsum(pay["jpeg_len"].numpy())])
        buf = pay["jpeg_buf"]

        def raw_img(i):
            return np.asarray(Image.open(io.BytesIO(bytes(buf[int(offs[i]):int(offs[i + 1])].numpy().tobytes())))
                              .convert("RGB")).astype(np.float32)
        mp4 = (sorted(glob.glob(f"{S}/frontwide/{cid}*.mp4")) + sorted(glob.glob(f"{S}/native7/{cid}*front_wide*.mp4")))
        tsp = (sorted(glob.glob(f"{S}/frontwide/{cid}*.timestamps.parquet"))
               + sorted(glob.glob(f"{S}/native7/{cid}*front_wide*.timestamps.parquet")))
        rec["mp4_found"] = [os.path.basename(x).replace(cid, "<clip>") for x in mp4]
        ts = pd.read_parquet(tsp[0])
        tcol = [c for c in ts.columns if "timestamp" in c][0]
        t_cam = ts[tcol].to_numpy().astype(np.int64)
        n_raw = int(len(offs) - 1)
        span = (t_cam[-1] - t_cam[0]) / 1e6
        t_q = np.linspace(t_cam[0], t_cam[-1], int(span * 10))
        rec["timing"] = {"n_cam_frames": int(t_cam.size), "n_query_linspace": int(t_q.size), "n_raw_cached": n_raw,
                         "timestamp_column": tcol}
        want = sorted(set(int(x) for x in np.linspace(5, min(n_raw, t_q.size) - 6, a.n_frames)))
        need = {}
        for k in want:
            j = int(np.searchsorted(t_cam, t_q[k]))
            for jj in range(j - 2, j + 3):
                need.setdefault(jj, []).append(k)
        frames = {}
        cont = av.open(mp4[0])
        for n_f, frm in enumerate(cont.decode(video=0)):
            if n_f in need:
                frames[n_f] = frm.to_ndarray(format="rgb24").astype(np.float32)
            if n_f > max(need):
                break
        dec_h, dec_w = next(iter(frames.values())).shape[:2]
        rec["decoded_hw"] = [dec_h, dec_w]
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        per = []
        for k in want:
            cached = raw_img(k).reshape(-1, 3)
            j0 = int(np.searchsorted(t_cam, t_q[k]))
            # 1) which mp4 frame IS this cached frame (at the declared geometry)?
            cand = {}
            ray0 = cyl_pixel_ray(uu, vv, W, H, F).reshape(-1, 3)
            un, vn = ftheta_from_cam(ray0, poly, icx, icy)
            un, vn = un * dec_w / iw, vn * dec_h / ih
            for jj in sorted(j_ for j_, ks in need.items() if k in ks and j_ in frames):
                smp, ins = bilinear(frames[jj], un, vn)
                m = ins & (ray0[:, 2] > 0) & (cached.sum(1) > 0)
                cand[jj - j0] = float(np.abs(smp[m] - cached[m]).mean())
            off = min(cand, key=cand.get)
            src = frames[j0 + off]
            # 2) geometry: shift the canonical grid by (du, dv) px and by a small ROLL; the minimum must be at 0
            res = {}
            for du in (-2, -1, 0, 1, 2):
                for dv in (-2, -1, 0, 1, 2):
                    ray = cyl_pixel_ray(uu + du, vv + dv, W, H, F).reshape(-1, 3)
                    un, vn = ftheta_from_cam(ray, poly, icx, icy)
                    smp, ins = bilinear(src, un * dec_w / iw, vn * dec_h / ih)
                    m = ins & (ray[:, 2] > 0) & (cached.sum(1) > 0)
                    res[(du, dv)] = float(np.abs(smp[m] - cached[m]).mean())
            fsc = {}
            for s in (0.99, 0.995, 1.0, 1.005, 1.01):                       # f_ref scale
                ray = cyl_pixel_ray(uu, vv, W, H, F * s).reshape(-1, 3)
                un, vn = ftheta_from_cam(ray, poly, icx, icy)
                smp, ins = bilinear(src, un * dec_w / iw, vn * dec_h / ih)
                m = ins & (ray[:, 2] > 0) & (cached.sum(1) > 0)
                fsc[s] = float(np.abs(smp[m] - cached[m]).mean())
            best = min(res, key=res.get)
            per.append({"cached_raw_frame": k, "mp4_offset_vs_linspace_searchsorted": off,
                        "mae_by_mp4_offset": {str(kk): round(v, 3) for kk, v in cand.items()},
                        "mae_at_declared_geometry": round(res[(0, 0)], 3), "best_shift_px": list(best),
                        "mae_best_shift": round(res[best], 3),
                        "mae_shift_grid": {f"{d_[0]},{d_[1]}": round(v, 3) for d_, v in res.items()},
                        "mae_by_f_scale": {str(s): round(v, 3) for s, v in fsc.items()},
                        "best_f_scale": min(fsc, key=fsc.get)})
        rec["per_frame"] = per
        rec["verdict"] = {"all_best_shift_zero": all(p["best_shift_px"] == [0, 0] for p in per),
                          "all_best_f_scale_1": all(p["best_f_scale"] == 1.0 for p in per)}
    else:
        sys.path.insert(0, "/home/nvidia/sam3map")
        import camera_model as CM
        cams = sorted(glob.glob(f"/home/nvidia/sam3map/native7/seq_{a.c8}/*/calib.npz"))
        c = np.load(cams[0], allow_pickle=True)
        rec["calib_npz_keys"] = list(c.files)
        rec["n_calib_npz"] = len(cams)
        idx = int(np.argmin(np.abs(np.asarray(c["ftheta_cx"]) - icx) + np.abs(np.asarray(c["ftheta_cy"]) - icy)))
        rec["front_wide_index_rule"] = {"idx": idx, "residual_px": float(abs(c["ftheta_cx"][idx] - icx)
                                                                       + abs(c["ftheta_cy"][idx] - icy))}
        scam = CM.Camera.from_calib(c, idx)
        R_cl, t_cl = Rl.T @ Rc, Rl.T @ (tc - tl)
        rec["sam3_calib_vs_parquet"] = {"R_vs_cam2rig": float(np.abs(scam.R - Rc).max()),
                                        "t_vs_cam_in_rig": float(np.abs(scam.t - tc).max()),
                                        "R_vs_cam2lidar": float(np.abs(scam.R - R_cl).max()),
                                        "t_vs_cam_in_lidar": float(np.abs(scam.t - t_cl).max()),
                                        "poly_vs_parquet": float(np.abs(np.asarray(scam.ft["poly"]) - np.asarray(poly)).max()),
                                        "cx_cy_vs_parquet": [float(scam.ft["cx"] - icx), float(scam.ft["cy"] - icy)]}
        rng = np.random.default_rng(0)
        P = np.c_[rng.uniform(3, 60, 20000), rng.uniform(-25, 25, 20000), rng.uniform(-0.5, 3.0, 20000)]
        u2, v2, ok2, Pc = cyl_project(P, Rc, tc, W, H, F)
        ray = cyl_pixel_ray(u2, v2, W, H, F)
        un, vn = ftheta_from_cam(ray, poly, icx, icy)
        for tag, PP in (("points_in_rig_frame", P), ("points_in_lidar_frame", (P - tl) @ Rl)):
            us, vs, oks = scam.project_rig(PP)
            both = oks & ok2
            rec[tag] = {"n": int(both.sum()),
                        "max_abs_du_native_px": float(np.abs(us[both] - un[both]).max()) if both.any() else None,
                        "max_abs_dv_native_px": float(np.abs(vs[both] - vn[both]).max()) if both.any() else None,
                        "p99_abs_dv_native_px": float(np.percentile(np.abs(vs[both] - vn[both]), 99)) if both.any() else None}
    json.dump(rec, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(rec, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
