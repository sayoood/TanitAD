"""Export the SAM3 map as per-frame BEV ground truth on the SAME grids the BEV head reads (label artifact only).

Consumer contract (TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/,
code/lidar_bev.py + code/bev_gt_loader.py): rig frame +x forward, +y LEFT, +z UP, origin rear axle on the road plane;
  cartesian  x in [0, 60) m, y in [-16, 16) m, 0.5 m cells, [120, 64]; row 0 at x in [0, 0.5); col 0 is y = -16 (RIGHT)
  polar      [24, 20] and polar48 [48, 40]: r in [0, 60) m, azimuth atan2(y, x) within +-60 deg, col 0 = +60 deg (LEFT)
  fine       x in [0, 60) m, y in [-16, 16) m at 0.10 m, [600, 320] class codes (this map's native resolution)
Each coarse cell carries SOFT class fractions from 5 x 5 sub-samples of the whole-clip world map (uint8, 255 = all):
channels 0 "seen, no map class", 1 drivable, 2 lane / road line, 3 crosswalk, 4 arrow / text, 5 non-drivable edge,
6 hatched area, 7 sidewalk / verge, 8 NOT SEEN by any camera in the clip. Channels sum to 255 (+-4 rounding).
The world map (sam3map_render_v5.py worldmap.npz) is NON-CAUSAL: a cell's class comes from the nearest camera view over
the whole clip. That is admissible for labels; an inference path must never read this file.
Usage: sam3map_export_gt.py <c8> <npz dir> <render dir with worldmap.npz> <out .npz>   (SAM3MAP_ROOT = sequence root)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

SCHEMA = "tanitad.sam3_map_gt/1"
CART = {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5, "shape": [120, 64]}
FINE = {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.1, "shape": [600, 320]}
POLAR = {"n_rng": 24, "n_az": 20, "r_max_m": 60.0, "hfov_deg": 120.0, "cell_deg": 6.0, "cell_rng_m": 2.5, "shape": [24, 20]}
POLAR48 = {"n_rng": 48, "n_az": 40, "r_max_m": 60.0, "hfov_deg": 120.0, "cell_deg": 3.0, "cell_rng_m": 1.25, "shape": [48, 40]}
SUB = 5
CLASS_NAMES = ["seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text", "non-drivable edge", "hatched area",
               "sidewalk / verge", "not seen"]


def lookup(wm, xy_world):
    """World-map codes at world points (255 outside the map)."""
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    i = np.floor((xy_world[:, 0] - x0) / res).astype(np.int64); j = np.floor((xy_world[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    out = np.full(len(xy_world), 255, np.uint8); out[ok] = cls[i[ok], j[ok]]
    return out


def fractions(codes, n_cells):
    """codes [n_cells * SUB^2] -> uint8 [9, n_cells] soft fractions (255 = not seen -> channel 8)."""
    ch = np.where(codes == 255, 8, codes).astype(np.int64).reshape(n_cells, SUB * SUB)
    cnt = np.stack([(ch == k).sum(axis=1) for k in range(9)])
    return np.round(cnt * (255.0 / (SUB * SUB))).astype(np.uint8)


def cart_points(spec):
    nx, ny = spec["shape"]; c = spec["cell_m"]
    o = (np.arange(SUB) + 0.5) / SUB
    ix, iy, sx, sy = np.meshgrid(np.arange(nx), np.arange(ny), o, o, indexing="ij")
    x = (ix + sx) * c; y = -spec["y_half_m"] + (iy + sy) * c
    return np.c_[x.ravel(), y.ravel()], nx * ny


def polar_points(spec):
    nr, na = spec["n_rng"], spec["n_az"]; half = spec["hfov_deg"] / 2
    o = (np.arange(SUB) + 0.5) / SUB
    ir, ia, sr, sa = np.meshgrid(np.arange(nr), np.arange(na), o, o, indexing="ij")
    r = (ir + sr) * spec["cell_rng_m"]; az = np.radians(half - (ia + sa) * spec["cell_deg"])      # col 0 = +half = LEFT
    return np.c_[(r * np.cos(az)).ravel(), (r * np.sin(az)).ravel()], nr * na


def main():
    c8, npz_dir, rdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    root = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
    t0 = time.time()
    wm = dict(np.load(rdir / "worldmap.npz", allow_pickle=True))
    if out.is_dir():
        out = out / f"{str(wm['clip_sha12'])}.sam3mapgt.npz"                  # sha12 names only (clip ids are gated)
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    Ts, toks = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            Ts.append(np.array(z["T_world_rig"])); toks.append(str(z["tok"])); ver = str(z["ver"]) if "ver" in z.files else "?"
    metas = [json.loads((root / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8")) for t in toks]
    grids = {"cart": (cart_points(CART), CART), "polar": (polar_points(POLAR), POLAR), "polar48": (polar_points(POLAR48), POLAR48)}
    fx = (np.arange(FINE["shape"][0]) + 0.5) * FINE["cell_m"]; fy = -FINE["y_half_m"] + (np.arange(FINE["shape"][1]) + 0.5) * FINE["cell_m"]
    FX, FY = np.meshgrid(fx, fy, indexing="ij"); fine_xy = np.c_[FX.ravel(), FY.ravel()]
    arrays = {f"{g}_frac": [] for g in grids}; arrays["fine_codes"] = []
    for T in Ts:
        R, t = T[:2, :2], T[:2, 3]
        for g, ((pts, n_cells), spec) in grids.items():
            arrays[f"{g}_frac"].append(fractions(lookup(wm, pts @ R.T + t), n_cells).reshape(9, *spec["shape"]))
        arrays["fine_codes"].append(lookup(wm, fine_xy @ R.T + t).reshape(FINE["shape"]))
    meta = {"schema": SCHEMA, "frame": "rig", "frame_convention": "+x forward, +y LEFT, +z UP; origin rear axle on the road plane",
            "cartesian": {**CART, "row0": "x in [0, cell_m); col0 is y = -y_half (RIGHT), +y is LEFT"},
            "polar": {**POLAR, "col0": "+hfov/2 = LEFT"}, "polar48": {**POLAR48, "col0": "+hfov/2 = LEFT"}, "fine": {**FINE, "codes": "0-7 class, 255 not seen"},
            "channels": CLASS_NAMES, "sub_samples_per_axis": SUB, "fraction_scale": 255,
            "source": {"sam3_map_version": ver, "world_map_res_m": float(wm["res"]), "renderer": "sam3map_render_v5 (nearest-view world map)",
                       "clip_sha12": str(wm["clip_sha12"])},
            "time_grid": "one row per extraction token (5 Hz, t_ref_us); the pose is the token's T_world_rig",
            "non_causal": "labels use every frame of the clip; inference must never read this file",
            "label_only": True}
    np.savez_compressed(out, meta_json=json.dumps(meta), t_ref_us=np.array([m["t_ref_us"] for m in metas], np.float64), T_world_rig=np.stack(Ts),
                        **{k: np.stack(v) for k, v in arrays.items()})
    # content assertions (a GT that is all "not seen" or all background must not pass silently)
    cf = np.stack(arrays["cart_frac"]).astype(np.float64)
    seen = 1 - cf[:, 8].mean() / 255; drivable = cf[:, 1].mean() / 255
    sums = cf.sum(axis=1); bad_sum = int((np.abs(sums - 255) > 5).sum())
    # ORIENTATION, from an independent source: the ego's own next 3 s of poses must lie on drivable ground in the fine
    # grid far more often than their mirror image (y -> -y) does; a mirrored or rotated export fails this.
    fine = np.stack(arrays["fine_codes"])
    hit = {"real": [0, 0], "mirror": [0, 0]}
    for n, T in enumerate(Ts):
        fut = np.array([Tm[:2, 3] for Tm in Ts[n + 1: n + 16]])
        if not len(fut):
            continue
        q = (fut - T[:2, 3]) @ T[:2, :2]
        for name, sgn in (("real", 1.0), ("mirror", -1.0)):
            i = np.floor(q[:, 0] / FINE["cell_m"]).astype(int); j = np.floor((sgn * q[:, 1] + FINE["y_half_m"]) / FINE["cell_m"]).astype(int)
            ok = (i >= 0) & (i < FINE["shape"][0]) & (j >= 0) & (j < FINE["shape"][1]) & (q[:, 0] >= 2.0)
            c = fine[n, i[ok], j[ok]]; c = c[c != 255]
            hit[name][0] += int(np.isin(c, (1, 2, 3, 4, 6)).sum()); hit[name][1] += len(c)
    path_on_road = {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in hit.items()}
    # two grids, one map: the polar48 drivable fraction at a cell centre must match the cartesian cell containing it
    # (same frame) much better than the cartesian cell of a frame 40 tokens away (control)
    p48 = np.stack(arrays["polar48_frac"]).astype(np.float64) / 255
    (ppts, _), _ = grids["polar48"]
    pc = ppts.reshape(POLAR48["n_rng"], POLAR48["n_az"], SUB * SUB, 2).mean(axis=2)
    ci = np.floor(pc[..., 0] / CART["cell_m"]).astype(int); cj = np.floor((pc[..., 1] + CART["y_half_m"]) / CART["cell_m"]).astype(int)
    inb = (ci >= 0) & (ci < CART["shape"][0]) & (cj >= 0) & (cj < CART["shape"][1])
    same, other = [], []
    for n in range(len(Ts)):
        a = p48[n, 1][inb]; b = cf[n, 1][ci[inb], cj[inb]] / 255; o = cf[(n + 40) % len(Ts), 1][ci[inb], cj[inb]] / 255
        same.append(np.abs(a - b).mean()); other.append(np.abs(a - o).mean())
    rep = {"frames": len(Ts), "cart_seen_share": round(float(seen), 4), "cart_drivable_share": round(float(drivable), 4),
           "cart_class_share": {CLASS_NAMES[k]: round(float(cf[:, k].mean() / 255), 4) for k in range(9)}, "cells_bad_sum": bad_sum,
           "ego_future_path_on_drivable": path_on_road, "polar48_vs_cart_drivable_mae": {"same_frame": round(float(np.mean(same)), 4),
                                                                                        "frame_plus_40_control": round(float(np.mean(other)), 4)},
           "bytes": out.stat().st_size, "seconds": round(time.time() - t0, 1)}
    assert seen > 0.2 and drivable > 0.05 and bad_sum == 0, rep
    assert path_on_road["real"] is not None and path_on_road["real"] >= 0.9 and path_on_road["real"] > path_on_road["mirror"], rep
    assert np.mean(same) < np.mean(other), rep
    (out.parent / (out.stem + ".report.json")).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep)); print("ZZEXPORTGT-OKZZ")


if __name__ == "__main__":
    main()
