"""Cross-camera registration of the ground warp, MEASURED from the paint itself (no LiDAR, no calibration target).

For each camera c: its bright-paint flags on the metric layer (cells within NEAR_M of the camera) are correlated with the
fused paint share of the OTHER cameras over the whole clip, at rig-frame shifts dx, dy in [-0.4, 0.4] m (0.1 m steps). A
well-registered camera peaks at (0, 0); a constant peak elsewhere is that camera's ground-warp error in the rig frame
(extrinsics, ground height, lens model), the same for every frame. Controls: the peak NCC must beat the NCC of a
MIRRORED others-map (y -> -y in the rig frame, same cells), and a camera's own map correlated with itself peaks at 0 by
construction (not reported). Evidence: MEASURED on the clip given. Usage: cross_cam_registration.py <c8> <npz dir> <out json>
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p_ in ("/home/nvidia/sam3paint", "/home/nvidia/sam3map", "/home/nvidia/sam3map/eval", str(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v5 as R5

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
RES, NEAR_M = R5.RES, 12.0
SH = int(os.environ.get("REG_SHIFT_CELLS", "8"))                      # +-0.8 m
REF = os.environ.get("REG_REF", "")                                  # e.g. CAM_FW: register every camera against that one only
FRAME_STRIDE = int(os.environ.get("REG_FRAME_STRIDE", "2"))


def main():
    c8, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))[::FRAME_STRIDE]
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    sd = ROOT / f"seq_{c8}"
    cams = [c for c in R5.R4.VIEWS if f"cls_{c}" in frames[0]]
    Ts = [f["T_world_rig"] for f in frames]; path = np.array([T[:2, 3] for T in Ts])
    wx0, wy0 = path.min(axis=0) - (R5.R_MAX + 12); wx1, wy1 = path.max(axis=0) + (R5.R_MAX + 12)
    WW, WH = int(np.ceil((wx1 - wx0) / RES)), int(np.ceil((wy1 - wy0) / RES))
    pnt = {c: np.zeros((WW, WH), np.float32) for c in cams}; obs = {c: np.zeros((WW, WH), np.float32) for c in cams}
    keep = []                                     # per (frame, camera): rig xy of near observed cells + paint flag
    for n, f in enumerate(frames):
        fd = sd / str(f["tok"])
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sg = GS.smooth_grid(grid, fb)
        for cam in cams:
            i = fr["cam_order"].index(cam)
            C = CM.Camera.from_calib(c, i)
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            code, _, paint, _ = R5.layer_fields(C, sg, img, f[f"cls_{cam}"], f.get(f"ego_{cam}"))
            # cells selected by the camera's OWN SAM3 paint classes (dilated 0.8 m) -- never by the others' paint, which would
            # condition the sample on the variable being correlated (selection bias: a negative NCC at zero shift, peaks at the edge)
            cand = ndi.binary_dilation(np.isin(code, (2, 3, 4, 6)), iterations=8) & (code != 255)
            gx = R5.LX[0] + (np.arange(R5.NX) + 0.5) * RES; gy = R5.LY[0] + (np.arange(R5.NY) + 0.5) * RES
            near_grid = np.hypot(gx[:, None] - C.t[0], gy[None, :] - C.t[1]) <= NEAR_M
            # the REFERENCE map: every observed cell within NEAR_M (paint share = bright cells / observations)
            li, lj = np.nonzero((code != 255) & near_grid)
            w = np.c_[R5.LX[0] + (li + 0.5) * RES, R5.LY[0] + (lj + 0.5) * RES] @ Ts[n][:2, :2].T + Ts[n][:2, 3]
            wi = np.floor((w[:, 0] - wx0) / RES).astype(np.int64); wj = np.floor((w[:, 1] - wy0) / RES).astype(np.int64)
            ok = (wi >= 0) & (wi < WW) & (wj >= 0) & (wj < WH)
            np.add.at(obs[cam], (wi[ok], wj[ok]), 1.0); np.add.at(pnt[cam], (wi[ok], wj[ok]), paint[li[ok], lj[ok]].astype(np.float32))
            # the SAMPLE correlated against the others: selected by this camera's own SAM3 paint classes only
            li, lj = np.nonzero(cand & near_grid)
            keep.append((n, cam, np.c_[R5.LX[0] + (li + 0.5) * RES, R5.LY[0] + (lj + 0.5) * RES], paint[li, lj]))
        if n % 10 == 0:
            print(f"  frame {n * FRAME_STRIDE}: {time.time() - t0:.0f}s", flush=True)
    shifts = [(dx * RES, dy * RES) for dx in range(-SH, SH + 1) for dy in range(-SH, SH + 1)]
    res = {}
    for cam in cams:
        refs = [REF] if (REF and cam != REF and REF in cams) else [c for c in cams if c != cam]
        po = sum(pnt[c] for c in refs); oo = sum(obs[c] for c in refs)
        share_o = np.where(oo >= 2, po / np.maximum(oo, 1), np.nan).astype(np.float32)
        acc = {k: np.zeros(len(shifts)) for k in ("ab", "a", "b", "aa", "bb", "n")}
        accm = {k: np.zeros(len(shifts)) for k in ("ab", "a", "b", "aa", "bb", "n")}
        for n, c_, xy, pf in keep:
            if c_ != cam or not len(xy):
                continue
            T = Ts[n]
            w0 = xy @ T[:2, :2].T + T[:2, 3]
            wi = np.floor((w0[:, 0] - wx0) / RES).astype(np.int64); wj = np.floor((w0[:, 1] - wy0) / RES).astype(np.int64)
            ok = (wi >= 0) & (wi < WW) & (wj >= 0) & (wj < WH)
            if len(xy) < 20:
                continue
            xy_s, a = xy, pf.astype(np.float64)
            for s_i, (dx, dy) in enumerate(shifts):
                for mirror, A in ((False, acc), (True, accm)):
                    q = xy_s + [dx, dy]
                    if mirror:
                        q = q * [1.0, -1.0]
                    w = q @ T[:2, :2].T + T[:2, 3]
                    wi = np.floor((w[:, 0] - wx0) / RES).astype(np.int64); wj = np.floor((w[:, 1] - wy0) / RES).astype(np.int64)
                    ok = (wi >= 0) & (wi < WW) & (wj >= 0) & (wj < WH)
                    b = np.full(len(q), np.nan); b[ok] = share_o[wi[ok], wj[ok]]
                    v = np.isfinite(b)
                    if not v.any():
                        continue
                    aa, bb = a[v], b[v]
                    A["ab"][s_i] += (aa * bb).sum(); A["a"][s_i] += aa.sum(); A["b"][s_i] += bb.sum()
                    A["aa"][s_i] += (aa * aa).sum(); A["bb"][s_i] += (bb * bb).sum(); A["n"][s_i] += len(aa)

        def ncc(A):
            nn = np.maximum(A["n"], 1)
            cov = A["ab"] / nn - (A["a"] / nn) * (A["b"] / nn)
            va = A["aa"] / nn - (A["a"] / nn) ** 2; vb = A["bb"] / nn - (A["b"] / nn) ** 2
            return cov / np.sqrt(np.maximum(va * vb, 1e-12))
        r, rm = ncc(acc), ncc(accm)
        k = int(np.argmax(r)); k0 = shifts.index((0.0, 0.0))
        grid_r = r.reshape(2 * SH + 1, 2 * SH + 1); pi, pj = divmod(k, 2 * SH + 1)
        around = grid_r[max(0, pi - 1): pi + 2, max(0, pj - 1): pj + 2]
        res[cam] = {"reference": refs, "peak_at_range_edge": bool(pi in (0, 2 * SH) or pj in (0, 2 * SH)), "ncc_3x3_around_peak": np.round(around, 3).tolist(),
                    "peak_shift_m": {"dx_forward": round(shifts[k][0], 2), "dy_left": round(shifts[k][1], 2)}, "peak_ncc": round(float(r[k]), 3),
                    "ncc_at_zero": round(float(r[k0]), 3), "mirrored_control_peak_ncc": round(float(rm.max()), 3), "cells_at_zero": int(acc["n"][k0])}
        print(cam, json.dumps(res[cam]), flush=True)
    rep = {"clip_frames_used": len(frames), "frame_stride": FRAME_STRIDE, "near_m": NEAR_M, "shift_range_m": SH * RES, "reference_camera": REF or "all others", "cameras": res,
           "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("ZZREG-DONEZZ")


if __name__ == "__main__":
    main()
