"""Timed /3 re-export PROTOTYPE at the census-selected extent (PREREG_MAP_EXTENT_CENSUS.md §7), on 20 TRAIN clips.

NOT the corpus exporter and writes NOTHING to disk except its own JSON record: every /3 array is built in memory,
compressed into an in-memory buffer (np.savez_compressed -> BytesIO) to time and size it, then dropped.

The /3 arrays = the /2 exporter's arrays at the new extent (sam3map_prod.export_v2ep + sam3map_export_gt, copied
verbatim below): fine_codes [T, X/0.1, 2Y/0.1], cart_frac [T, 9, X/0.5, 2Y/0.5] (5 x 5 sub-samples), and the polar /
polar48 grids UNCHANGED (their definitions are range 60 m, +-60 deg; the SPEC does not move them).

⭐ ANCHORED COORDINATES -- the only way the SPEC's control can hold. The /2 exporter writes the lateral cell centre as
`-y_half + (j + 0.5) * cell`. Written that way at y_half = 30, a cell inside the old window gets a DIFFERENT float than
/2 gave it (-30 + (j + 140.5) * 0.1 != -16 + (j + 0.5) * 0.1 in general), and a floor() at a world-map cell boundary can
then flip a code. So /3 writes every coordinate relative to the /2 window's origin: y = -16 + (j_rel + 0.5) * cell with
j_rel = j - (Y_new - 16) / cell. Inside the old window j_rel IS the /2 index, so the arithmetic is character-identical.
Both identities are CHECKED on every frame of the 20 clips: fine_codes and cart_frac inside the old window vs /2.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/nvidia/sam3map/corpus/out")
CLIPS = Path("/home/nvidia/data/refcv6_train_clips.txt")
SUB = 5
POLAR = {"n_rng": 24, "n_az": 20, "r_max_m": 60.0, "hfov_deg": 120.0, "cell_deg": 6.0, "cell_rng_m": 2.5, "shape": [24, 20]}
POLAR48 = {"n_rng": 48, "n_az": 40, "r_max_m": 60.0, "hfov_deg": 120.0, "cell_deg": 3.0, "cell_rng_m": 1.25, "shape": [48, 40]}


def lookup(wm, xy_world):                                   # sam3map_export_gt.py:30-36, verbatim
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    i = np.floor((xy_world[:, 0] - x0) / res).astype(np.int64); j = np.floor((xy_world[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    out = np.full(len(xy_world), 255, np.uint8); out[ok] = cls[i[ok], j[ok]]
    return out


def fractions(codes, n_cells):                              # sam3map_export_gt.py:39-43, verbatim
    ch = np.where(codes == 255, 8, codes).astype(np.int64).reshape(n_cells, SUB * SUB)
    cnt = np.stack([(ch == k).sum(axis=1) for k in range(9)])
    return np.round(cnt * (255.0 / (SUB * SUB))).astype(np.uint8)


def polar_points(spec):                                     # sam3map_export_gt.py:56-62, verbatim
    nr, na = spec["n_rng"], spec["n_az"]; half = spec["hfov_deg"] / 2
    o = (np.arange(SUB) + 0.5) / SUB
    ir, ia, sr, sa = np.meshgrid(np.arange(nr), np.arange(na), o, o, indexing="ij")
    r = (ir + sr) * spec["cell_rng_m"]; az = np.radians(half - (ia + sa) * spec["cell_deg"])
    return np.c_[(r * np.cos(az)).ravel(), (r * np.sin(az)).ravel()], nr * na


def cart_points_anchored(x_max, y_half, c=0.5, y_old=16.0):
    """cart_points (export_gt.py:46-53) at a new extent, lateral index relative to the /2 window."""
    nx, ny = int(round(x_max / c)), int(round(2 * y_half / c))
    off = int(round((y_half - y_old) / c))
    o = (np.arange(SUB) + 0.5) / SUB
    ix, iy, sx, sy = np.meshgrid(np.arange(nx), np.arange(ny) - off, o, o, indexing="ij")
    x = (ix + sx) * c; y = -y_old + (iy + sy) * c
    return np.c_[x.ravel(), y.ravel()], nx * ny, (nx, ny), off


def fine_xy_anchored(x_max, y_half, c=0.1, y_old=16.0):
    nx, ny = int(round(x_max / c)), int(round(2 * y_half / c))
    off = int(round((y_half - y_old) / c))
    fx = (np.arange(nx) + 0.5) * c; fy = -y_old + (np.arange(ny) - off + 0.5) * c
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.c_[FX.ravel(), FY.ravel()], (nx, ny), off


def sha12(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-max", type=float, required=True)
    ap.add_argument("--y-half", type=float, required=True)
    ap.add_argument("--n-clips", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = [l.strip() for l in open(CLIPS) if l.strip()]
    s12s = sorted({sha12(c) for c in clips})[: a.n_clips]
    fxy, fshape, foff = fine_xy_anchored(a.x_max, a.y_half)
    (cpts, cn, cshape, coff) = cart_points_anchored(a.x_max, a.y_half)
    grids = {"polar": polar_points(POLAR), "polar48": polar_points(POLAR48)}
    rows, t_all = [], time.time()
    for s12 in s12s:
        t0 = time.time()
        wm = dict(np.load(OUT_DIR / f"{s12}.worldmap.npz", allow_pickle=True))
        with np.load(OUT_DIR / f"{s12}.sam3mapgt.npz", allow_pickle=False) as z:
            Ts = z["T_world_rig"]; fine2 = z["fine_codes"]; cart2 = z["cart_frac"]
            t_q, t_i, fidx = z["t_query_us"], z["t_img_us"], z["cam_frame_idx"]
        t_load = time.time() - t0
        fine3, cart3, pol = [], [], {k: [] for k in grids}
        for T in Ts:
            Rm, t = T[:2, :2], T[:2, 3]
            fine3.append(lookup(wm, fxy @ Rm.T + t).reshape(fshape))
            cart3.append(fractions(lookup(wm, cpts @ Rm.T + t), cn).reshape(9, *cshape))
            for g, (pts, n) in grids.items():
                pol[g].append(fractions(lookup(wm, pts @ Rm.T + t), n).reshape(9, *(POLAR if g == "polar" else POLAR48)["shape"]))
        fine3 = np.stack(fine3); cart3 = np.stack(cart3)
        t_compute = time.time() - t0 - t_load
        # the SPEC's control: /3 inside the old window == /2, byte for byte (every frame)
        fin_in = fine3[:, :600, foff:foff + 320]
        car_in = cart3[:, :, :120, coff:coff + 64]
        d_fine = int((fin_in != fine2).sum()); d_cart = int((car_in != cart2).sum())
        t1 = time.time()
        buf = io.BytesIO()
        np.savez_compressed(buf, meta_json=json.dumps({"schema": "tanitad.sam3_map_gt/3 (PROTOTYPE, not written)"}),
                            t_query_us=t_q, t_img_us=t_i, cam_frame_idx=fidx, T_world_rig=Ts, fine_codes=fine3,
                            cart_frac=cart3, **{f"{g}_frac": np.stack(v) for g, v in pol.items()})
        t_save = time.time() - t1
        rows.append({"sha12": s12, "T": int(len(Ts)), "load_s": round(t_load, 2), "compute_s": round(t_compute, 2),
                     "compress_s": round(t_save, 2), "total_s": round(time.time() - t0, 2),
                     "bytes_v3_compressed": buf.getbuffer().nbytes,
                     "bytes_v2_file": (OUT_DIR / f"{s12}.sam3mapgt.npz").stat().st_size,
                     "old_window_fine_diff_bytes": d_fine, "old_window_cart_diff_bytes": d_cart,
                     "fine_shape": list(fine3.shape), "cart_shape": list(cart3.shape)})
        print(json.dumps(rows[-1]), flush=True)
        del buf, fine3, cart3, pol
    tot = [r["total_s"] for r in rows]
    rec = {"extent": {"x_max_m": a.x_max, "y_half_m": a.y_half}, "n_clips": len(rows),
           "anchored_offsets": {"fine_lateral_cells": foff, "cart_lateral_cells": coff},
           "per_clip_total_s": {"mean": round(float(np.mean(tot)), 2), "median": round(float(np.median(tot)), 2),
                                "max": round(float(np.max(tot)), 2)},
           "per_clip_bytes_v3_mean_MB": round(float(np.mean([r["bytes_v3_compressed"] for r in rows])) / 1e6, 2),
           "per_clip_bytes_v2_mean_MB": round(float(np.mean([r["bytes_v2_file"] for r in rows])) / 1e6, 2),
           "old_window_identity": {"fine_all_identical": all(r["old_window_fine_diff_bytes"] == 0 for r in rows),
                                   "cart_all_identical": all(r["old_window_cart_diff_bytes"] == 0 for r in rows),
                                   "n_frames_checked": int(sum(r["T"] for r in rows))},
           "wall_s": round(time.time() - t_all, 1), "rows": rows,
           "python": sys.version.split()[0], "numpy": np.__version__}
    n_all = 4369 + 139
    for w in (1, 4, 6, 8):
        rec[f"full_reexport_estimate_h_{w}_workers"] = round(rec["per_clip_total_s"]["mean"] * n_all / w / 3600, 2)
    rec["full_reexport_estimate_GB"] = round(rec["per_clip_bytes_v3_mean_MB"] * n_all / 1000, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(Path(a.out) / "reexport_v3_timing.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), flush=True)
    print("ZZREEXPORT-DONEZZ", flush=True)


if __name__ == "__main__":
    main()
