"""Why the night map loses its lane lines: per camera layer (the renderer's own metric layer, 0.1 m), the white top-hat of the
MIP luminance (1.5 m disc) on SAM3 line / arrow / hatched / crosswalk cells vs on drivable cells, and how many cells pass the
renderer's paint rule thr = max(FLOOR, K x MAD(road top-hat)) for several FLOOR / K. Road pass rate = false paint on asphalt;
thin-class pass rate = kept paint (SAM3 masks are wider than the paint, so it is an upper bound on precision, not recall).
Also split by camera range (0-10 / 10-20 / 20-35 m). Day clip beside night.
Usage: night_paint_probe.py <out json> <c8>=<npz dir> [<c8>=<npz dir> ...]   (every 8th frame)"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v5c as R

ROOT = Path("/home/nvidia/sam3map/native7")
RULES = [(10, 2.5), (6, 2.5), (4, 2.5), (4, 4.0), (3, 4.0), (2, 5.0)]
BANDS = ((0, 10), (10, 20), (20, 35))
out = Path(sys.argv[1]); rep = {}
for arg in sys.argv[2:]:
    c8, nd = arg.split("=", 1)
    files = sorted(Path(nd).glob("[0-9][0-9][0-9].npz"))[::8]
    acc = {}
    for f in files:
        z = np.load(f, allow_pickle=True); tok = str(z["tok"]); fd = ROOT / f"seq_{c8}" / tok
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
        for cam in [k[4:] for k in z.files if k.startswith("cls_")]:
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.shape[1], img.shape[0])
            # the renderer's layer, re-derived to expose the top-hat values (same code path as R.layer_fields)
            N = R.NX * R.NY
            gx, gy = np.meshgrid(R.GXL, R.GYL, indexing="ij")
            near = np.hypot(gx - C.t[0], gy - C.t[1]) <= R.R_MAX
            idx = np.flatnonzero(near.ravel()); xy = np.c_[gx.ravel()[idx], gy.ravel()[idx]]
            Pk = np.c_[xy, GS.height(xy, sg)]
            u, v, ok = C.project_rig(Pk); keep = np.flatnonzero(ok)
            idx, Pk, u, v = idx[keep], Pk[keep], u[keep], v[keep]
            if len(idx) < 100:
                continue
            cls_small = z[f"cls_{cam}"]; ego = z[f"ego_{cam}"] if f"ego_{cam}" in z.files else None
            su = np.clip((u / 2).astype(np.int32), 0, cls_small.shape[1] - 1); sv = np.clip((v / 2).astype(np.int32), 0, cls_small.shape[0] - 1)
            cl = cls_small[sv, su].copy()
            if ego is not None:
                cl[ego[sv, su]] = 255
            dist = np.hypot(Pk[:, 0] - C.t[0], Pk[:, 1] - C.t[1])
            ux, vx, _ = C.project_rig(Pk + [R.RES, 0.0, 0.0]); uy, vy, _ = C.project_rig(Pk + [0.0, R.RES, 0.0])
            fp = np.maximum(np.hypot(ux - u, vx - v), np.hypot(uy - u, vy - v))
            lvl = np.clip(np.floor(np.log2(np.maximum(fp, 1.0))), 0, R.MIP_LEVELS - 1).astype(np.int8)
            pyr = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)]
            for _ in range(R.MIP_LEVELS - 1):
                pyr.append(cv2.pyrDown(pyr[-1]))
            lum = np.zeros(len(idx), np.float32)
            for l in range(R.MIP_LEVELS):
                m = lvl == l
                if m.any():
                    s = 2.0 ** l
                    lum[m] = R.bilinear(pyr[l], (u[m] + 0.5) / s - 0.5, (v[m] + 0.5) / s - 0.5)
            road = cl == 1
            if road.sum() < 50:
                continue
            fill = float(np.median(lum[road]))
            L = np.full(N, fill, np.float32); L[idx] = lum
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(R.TOPHAT_M / R.RES)) | 1,) * 2)
            th = cv2.morphologyEx(L.reshape(R.NX, R.NY), cv2.MORPH_TOPHAT, se).ravel()[idx]
            mad = float(np.median(np.abs(th[road] - np.median(th[road]))))
            for a, b in BANDS:
                band = (dist >= a) & (dist < b)
                for name, sel in (("road", road & band), ("line", (cl == 2) & band), ("crosswalk", (cl == 3) & band), ("hatched_arrow", np.isin(cl, (4, 6)) & band)):
                    k_ = acc.setdefault(f"{cam} {a}-{b}m {name}", {"cells": 0, "lum_sum": 0.0, "th_p50": [], "th_p90": [], **{f"pass_f{fl}_k{kk}": 0 for fl, kk in RULES}})
                    if not sel.any():
                        continue
                    k_["cells"] += int(sel.sum()); k_["lum_sum"] += float(lum[sel].sum())
                    k_["th_p50"].append(float(np.percentile(th[sel], 50))); k_["th_p90"].append(float(np.percentile(th[sel], 90)))
                    for fl, kk in RULES:
                        k_[f"pass_f{fl}_k{kk}"] += int((th[sel] > max(fl, kk * mad)).sum())
            acc.setdefault(f"{cam} road_mad", []).append(mad)
    res = {}
    for k, v in acc.items():
        if isinstance(v, list):
            res[k] = round(float(np.median(v)), 2); continue
        if v["cells"] < 200:
            continue
        res[k] = {"cells": v["cells"], "lum_mean": round(v["lum_sum"] / v["cells"], 1), "th_p50_med": round(float(np.median(v["th_p50"])), 1),
                  "th_p90_med": round(float(np.median(v["th_p90"])), 1), **{r: round(v[r] / v["cells"], 3) for r in v if r.startswith("pass_")}}
    rep[c8] = res
    print(c8)
    for k, v in res.items():
        print("  ", k, v, flush=True)
out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("ZZPAINTPROBE-DONEZZ")
