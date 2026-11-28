"""SAM3 map-extent coverage census -- executes raw/PREREG_MAP_EXTENT_CENSUS.md (registered 2026-09-27 00:19, md5 in
the RESULT) and nothing else. Runs ON THOR: /home/nvidia/venvs/tanitad-edge/bin/python (the production interpreter),
CPU, nice 19, <= 6 workers. READ-ONLY on /home/nvidia/sam3map and /home/nvidia/data; writes only --out.

The exporter's arithmetic is COPIED here verbatim (sam3map_export_gt.lookup / the fine-grid expression of
sam3map_prod.export_v2ep) instead of imported, so that importing it cannot write a __pycache__ into the corpus tree.

Order: C1 + C2 (the validity gate) -> the census over every TRAIN clip -> the mechanical selection. If C1 or C2 is not
24/24 byte-identical the process writes the control record and EXITS 2 without computing a single coverage number.
Clip ids never leave the process: sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/nvidia/sam3map/corpus/out")
CLIPS = Path("/home/nvidia/data/refcv6_train_clips.txt")
# census rectangle: x in [0, 200) ahead, y in [-60, 60); samples at the 10 cm cell centres, 1 in 5 per axis.
NA, NB = 400, 240                          # x samples (0.5 m), y samples (0.5 m)
I_IDX = 5 * np.arange(NA) + 2              # 10 cm x index (anchored at x = 0, the exporter's own x formula)
J_IDX = 5 * np.arange(NB) + 2 - 440        # 10 cm y index RELATIVE TO THE /2 WINDOW (j = 0 is y = -16 + 0.05)
Y_OLD = 16.0
THRESH_RULE = 0.20
THRESHES = (0.05, 0.20, 0.50)


# ------------------------------------------------------------------ the exporter, verbatim (sam3map_export_gt.py:30-36)
def lookup(wm, xy_world):
    """World-map codes at world points (255 outside the map)."""
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    i = np.floor((xy_world[:, 0] - x0) / res).astype(np.int64); j = np.floor((xy_world[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    out = np.full(len(xy_world), 255, np.uint8); out[ok] = cls[i[ok], j[ok]]
    return out


def exporter_fine_xy():
    """sam3map_prod.export_v2ep lines 161-162, verbatim (X.FINE = x_max 60, y_half 16, 0.1 m, [600, 320])."""
    fx = (np.arange(600) + 0.5) * 0.1; fy = -16.0 + (np.arange(320) + 0.5) * 0.1
    FXg, FYg = np.meshgrid(fx, fy, indexing="ij")
    return np.c_[FXg.ravel(), FYg.ravel()]


def census_xy():
    """The census samples, with the SAME expression form as the exporter, anchored at the /2 window's origin:
    x = (i + 0.5) * 0.1, y = -16 + (j + 0.5) * 0.1 -- so a sample inside the /2 window has bit-identical coordinates
    to the exporter's cell (i, j)."""
    fx = (I_IDX + 0.5) * 0.1; fy = -16.0 + (J_IDX + 0.5) * 0.1
    FXg, FYg = np.meshgrid(fx, fy, indexing="ij")
    return np.c_[FXg.ravel(), FYg.ravel()], fx, fy


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def census_frames(T: int) -> list:
    return [r for r in range(9, T - 21, 10)]                       # r = 9 + 10k, r <= T - 22


def load_clip(s12: str):
    wmp, gtp = OUT_DIR / f"{s12}.worldmap.npz", OUT_DIR / f"{s12}.sam3mapgt.npz"
    if not wmp.is_file() or not gtp.is_file():
        return None
    wm = dict(np.load(wmp, allow_pickle=True))
    with np.load(gtp, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        Ts = z["T_world_rig"]
    if meta.get("schema") != "tanitad.sam3_map_gt/2" or (meta.get("source") or {}).get("clip_sha12") != s12:
        raise SystemExit(f"[census] {s12}: /2 schema or identity mismatch")
    if str(wm.get("clip_sha12")) != s12:
        raise SystemExit(f"[census] {s12}: world-map identity mismatch")
    return wm, Ts, gtp


# ------------------------------------------------------------------ rings
def ring_masks(fx, fy):
    ax = np.abs(fy)
    m = {}
    for k in range(20):
        xs = (fx >= 10 * k) & (fx < 10 * k + 10)
        m[f"x{k}|y<16"] = np.outer(xs, ax < 16.0)
        m[f"x{k}|y<10"] = np.outer(xs, ax < 10.0)
    for mm in range(6):
        ys = (ax >= 10 * mm) & (ax < 10 * mm + 10)
        m[f"y{mm}|x<60"] = np.outer(fx < 60.0, ys)
        m[f"y{mm}|x<30"] = np.outer(fx < 30.0, ys)
        for k in range(20):
            m[f"B{k},{mm}"] = np.outer((fx >= 10 * k) & (fx < 10 * k + 10), ys)
    return m


_CACHE = {}


def _geometry():
    """Per-process cache: the sample points and the ring index sets (identical in every process)."""
    if not _CACHE:
        pts, fx, fy = census_xy()
        masks = ring_masks(fx, fy)
        keys = list(masks)
        _CACHE.update(pts=pts, keys=keys, idx={k: np.flatnonzero(masks[k].ravel()) for k in keys})
    return _CACHE["pts"], _CACHE["keys"], _CACHE["idx"]


def worker(s12: str) -> dict:
    got = load_clip(s12)
    if got is None:
        return {"sha12": s12, "missing": True}
    wm, Ts, _ = got
    pts, keys, idx = _geometry()
    frames = census_frames(len(Ts))
    fr = np.zeros((len(frames), len(keys)), np.float32)
    for n, r in enumerate(frames):
        T = Ts[r]
        Rm, t = T[:2, :2], T[:2, 3]
        seen = lookup(wm, pts @ Rm.T + t) != 255
        for q, k in enumerate(keys):
            fr[n, q] = seen[idx[k]].mean()
    return {"sha12": s12, "missing": False, "T": int(len(Ts)), "n_frames": len(frames), "keys": keys,
            "frac": fr}


def controls(s12s: list) -> dict:
    """C1: the exporter's full fine grid re-cropped == stored /2 fine_codes, byte for byte.
    C2: the census sampler inside the /2 window == stored fine_codes[2::5, 2::5], byte for byte."""
    fxy = exporter_fine_xy()
    pts, fx, fy = census_xy()
    ia = np.flatnonzero((I_IDX >= 0) & (I_IDX < 600))            # census x samples inside the old window
    jb = np.flatnonzero((J_IDX >= 0) & (J_IDX < 320))
    rec = {"frames": [], "c1_identical": 0, "c2_identical": 0, "n": 0}
    for s12 in s12s:
        got = load_clip(s12)
        if got is None:
            continue
        wm, Ts, gtp = got
        with np.load(gtp, allow_pickle=False) as z:
            fine = z["fine_codes"]
        frames = census_frames(len(Ts))
        for r in (frames[0], frames[len(frames) // 2]):
            T = Ts[r]
            Rm, t = T[:2, :2], T[:2, 3]
            rec1 = lookup(wm, fxy @ Rm.T + t).reshape(600, 320)
            d1 = int((rec1 != fine[r]).sum())
            samp = lookup(wm, pts @ Rm.T + t).reshape(NA, NB)[np.ix_(ia, jb)]
            ref = fine[r][2::5, 2::5]
            d2 = int((samp != ref).sum()) if samp.shape == ref.shape else -1
            rec["frames"].append({"sha12": s12, "raw_frame": int(r), "c1_diff_bytes": d1, "c2_diff_bytes": d2,
                                  "c2_shape": list(samp.shape), "seen_share": float((fine[r] != 255).mean())})
            rec["c1_identical"] += int(d1 == 0)
            rec["c2_identical"] += int(d2 == 0)
            rec["n"] += 1
    rec["pass"] = bool(rec["n"] >= 24 and rec["c1_identical"] == rec["n"] and rec["c2_identical"] == rec["n"])
    return rec


def select(cov: dict, prefix: str, n: int, floor: float, step: float = 10.0) -> dict:
    K = -1
    for k in range(n):
        if cov[f"{prefix}{k}"] >= 0.50:
            K = k
        else:
            break
    edge = K == n - 1
    val = step * (K + 1)
    return {"last_passing_ring": K, "value_m": val, "census_edge_reached": edge,
            "after_floor_m": max(val, floor)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clips = [l.strip() for l in open(CLIPS) if l.strip()]
    s12s = sorted({sha12(c) for c in clips})
    rec = {"prereg": "raw/PREREG_MAP_EXTENT_CENSUS.md", "n_train_clips_listed": len(clips), "n_sha12": len(s12s),
           "numpy": np.__version__, "python": sys.version.split()[0], "interpreter": sys.executable}
    # ---- the validity gate first --------------------------------------------------------------------------------
    ctl = controls(s12s[:12])
    rec["controls"] = ctl
    json.dump(rec, open(out / "census_controls.json", "w"), indent=1)
    print(f"[census] controls: C1 {ctl['c1_identical']}/{ctl['n']}, C2 {ctl['c2_identical']}/{ctl['n']}, "
          f"pass {ctl['pass']} ({time.time() - t0:.1f} s)", flush=True)
    if not ctl["pass"]:
        print("[census] INVALID: the controls are not byte-identical -- no coverage number is computed", flush=True)
        sys.exit(2)
    # ---- the census ---------------------------------------------------------------------------------------------
    with Pool(int(a.workers)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(worker, s12s, chunksize=8)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(f"[census] {i + 1}/{len(s12s)} clips ({time.time() - t0:.0f} s)", flush=True)
    miss = [r["sha12"] for r in res if r["missing"]]
    ok = [r for r in res if not r["missing"]]
    rec["n_clips_missing"] = len(miss)
    rec["missing_sha12_first20"] = sorted(miss)[:20]
    rec["n_clips_used"] = len(ok)
    rec["inconclusive"] = len(miss) > 0.01 * len(s12s)
    keys = ok[0]["keys"]
    F = np.concatenate([np.asarray(r["frac"], np.float32) for r in ok])            # [frames, keys]
    per_clip_n = [r["n_frames"] for r in ok]
    rec["n_frames"] = int(F.shape[0])
    rec["frames_per_clip"] = {"min": int(min(per_clip_n)), "median": float(np.median(per_clip_n)),
                              "max": int(max(per_clip_n))}
    table = {}
    for q, k in enumerate(keys):
        col = F[:, q]
        row = {"mean_seen_frac": round(float(col.mean()), 4)}
        for th in THRESHES:
            row[f"cov@{th:.2f}"] = round(float((col >= th).mean()), 4)
        # clip-weighted coverage at the rule's threshold
        off, cw = 0, []
        for r in ok:
            c = col[off: off + r["n_frames"]]
            off += r["n_frames"]
            if len(c):
                cw.append(float((c >= THRESH_RULE).mean()))
        row["cov@0.20_clip_weighted"] = round(float(np.mean(cw)), 4)
        table[k] = row
    rec["table"] = table
    cov_x = {f"x{k}": table[f"x{k}|y<16"]["cov@0.20"] for k in range(20)}
    cov_y = {f"y{m}": table[f"y{m}|x<60"]["cov@0.20"] for m in range(6)}
    rec["rule_rings"] = {"x_rings_|y|<16_cov@0.20": cov_x, "y_rings_x<60_cov@0.20": cov_y}
    sx = select(cov_x, "x", 20, 60.0)
    sy = select(cov_y, "y", 6, 16.0)
    rec["selection"] = {"x_max": sx, "y_half": sy,
                        "extent": {"x_max_m": sx["after_floor_m"], "y_half_m": sy["after_floor_m"]},
                        "rule": "largest contiguous 10 m step with >= 50 % of census frames seen (>= 20 % of the ring's "
                                "samples not-255); floor 60 m x +-16 m"}
    # sensitivity (reported, not read by the rule)
    sens = {}
    for th in THRESHES:
        cx = {f"x{k}": table[f"x{k}|y<16"][f"cov@{th:.2f}"] for k in range(20)}
        cy = {f"y{m}": table[f"y{m}|x<60"][f"cov@{th:.2f}"] for m in range(6)}
        sens[f"thresh_{th:.2f}"] = {"x_max": select(cx, "x", 20, 60.0)["value_m"], "y_half": select(cy, "y", 6, 16.0)["value_m"]}
    cx10 = {f"x{k}": table[f"x{k}|y<10"]["cov@0.20"] for k in range(20)}
    cy30 = {f"y{m}": table[f"y{m}|x<30"]["cov@0.20"] for m in range(6)}
    sens["x_rings_within_|y|<10"] = select(cx10, "x", 20, 60.0)["value_m"]
    sens["y_rings_within_x<30"] = select(cy30, "y", 6, 16.0)["value_m"]
    cxw = {f"x{k}": table[f"x{k}|y<16"]["cov@0.20_clip_weighted"] for k in range(20)}
    cyw = {f"y{m}": table[f"y{m}|x<60"]["cov@0.20_clip_weighted"] for m in range(6)}
    sens["clip_weighted"] = {"x_max": select(cxw, "x", 20, 60.0)["value_m"], "y_half": select(cyw, "y", 6, 16.0)["value_m"]}
    rec["sensitivity_not_rule"] = sens
    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(out / "census_table.json", "w"), indent=1)
    print(json.dumps({"n_clips_used": rec["n_clips_used"], "n_missing": rec["n_clips_missing"], "n_frames": rec["n_frames"],
                      "x_rings": cov_x, "y_rings": cov_y, "selection": rec["selection"]["extent"],
                      "sensitivity": sens, "elapsed_s": rec["elapsed_s"]}), flush=True)
    print("ZZCENSUS-DONEZZ", flush=True)


if __name__ == "__main__":
    main()
