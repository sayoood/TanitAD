"""e0d_score.py -- E0d: re-score a BANKED map-raster bank (refcv8 EMA-48,852 by default) under the PUBLISHED thin-class protocol.

ZERO training, ZERO GPU, CPU only.  Reads `<rast>/e###_w####.npz` (planes psw pdrv pedge code gt; 1000 x 600 cells of 10 cm, ego frame at t0,
x forward 0..100 m, y -30..+30 m) and the GT planes of the E0h cache.  Implements REFCV9_DESIGN 4.1 row E0d / 5.2 rows MAP-3, MAP-4.

    python e0d_score.py selftest                         # analytic worlds only, reads no eval raster
    python e0d_score.py extract --out <stats.npz> ...    # pass 1: per-window sufficient statistics (DEV episodes only; detached)
    python e0d_score.py read --stats <stats.npz> --out-dir <raw dir>     # pass 2: thresholds on FIT, score on SCORE, bootstrap, controls

THE PROTOCOL (declared here BEFORE any statistic was read):
  * Split: DEV episodes of `DEV_HOLD_SPLIT.json` only (REFCV9_DESIGN 5.1: DEV selects, HOLD confirms).  HOLD is never opened (`episodes_for`
    refuses it unless E0D_UNSEAL_HOLD=yes, the MM's sealing step).  Inside DEV, in the file's own rank order (sha256(clip_id) ascending),
    even position = FIT, odd position = SCORE.  Thresholds are fitted on FIT counts ONLY and read on SCORE; the swapped fold and the 2-fold
    cross-fit over all DEV are reported BESIDE.
  * Range bands (rows of 10 cm): 0-20 / 20-40 / 40-60 / 60-100 m (stored at 20 m granularity so the perc8 five-band table can be compared).
  * Valid cells: GT != 255 (not seen).  `nocls` (0) is a NEGATIVE for the drivable union (perc8 convention); the J variant (nocls ignored) is beside.
  * MAP-4  drivable-union IoU: pdrv >= t vs GT in {1,2,3,4,6}; POOLED counts over windows; t = argmax pooled IoU on FIT, per band (primary) / pooled
    over bands (beside); fixed t = 128 (the declared prior_corrected 0.502 gate) beside.
  * MAP-3a lane tolerance-IoU (0.5 m): the lane plane is `code == 2` (NO lane probability plane is banked -> no lane threshold can be fitted;
    the lane is scored at the DECLARED operating point).  PRIMARY L-DIL = both planes rendered 0.5 m wide (square dilation 2 cells, 1 px -> 5 px) then
    pixel IoU, pooled.  Beside: exact 10 cm IoU, and the project's boundary-tolerant IoU (F/(2-F), pooled P_k and R_k) at k = 2 (0.2 m) and k = 5 (0.5 m).
  * MAP-3b drivable-boundary recall of GT edges @0.3 m: boundary of {pdrv >= t} (inner 8-neighbourhood, outer ring cleared) dilated 3 cells vs GT
    class 5; per-window recall over windows with >= 20 GT edge cells in the band, window-mean (pooled window mean); t = argmax pooled F@0.3 m on FIT.
    Fixed t = 128 (the H-R9-MAPEDGE-1 protocol) beside; its reproduction on DEV is a control.
  * Estimator: taniteval/ci.py episode-cluster bootstrap (`stat_cluster_bootstrap` / `paired_stat_cluster_bootstrap`), B 2000, seed 0, over EPISODES
    ("another draw of episodes" ONLY).  One training seed; the map head is deterministic given the frames.  Tier: perception instrument, map head
    vs SAM3 GT raster at the label level -- not a planning or driving metric.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------------------------------------------------------------------- #
H, W = 1000, 600
EDGES20 = (0, 200, 400, 600, 800, 1000)                      # five 20-m bands, rows of 10 cm
BAND4 = {"0-20": (0,), "20-40": (1,), "40-60": (2,), "60-100": (3, 4)}
BAND5 = {"0-20": (0,), "20-40": (1,), "40-60": (2,), "60-80": (3,), "80-100": (4,)}
DRV_CODES = (1, 2, 3, 4, 6)
NOCLS, LANE, EDGE, UNSEEN = 0, 2, 5, 255
TOL_EDGE = 3                                                  # 0.3 m
LANE_RS = (0, 2, 5)                                           # exact / 0.5 m rendering / 0.5 m tolerance (cells)
LANE_PRIMARY_R = 2
MIN_CELLS = 20
THR_DRV = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224)
THR_EDGE = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64)
FIXED_T = 128
N_BOOT = 2000
NAME = re.compile(r"e(\d+)_w(\d+)\.npz$")
# published references (INHERITED, not re-measured here): H-R9-MAPEDGE-1 and perc8, 4,686 windows / 137 episodes, all of eval139
PUB_MAPEDGE = {"rec": (0.29069, 0.26709, 0.31469), "rec_0-20m": (0.41162, 0.38395, 0.44023), "gt_bnd_rec": 0.9641}
PUB_PERC8 = {"drivable": [0.605, 0.613, 0.548, 0.517, 0.490], "lane": [0.199, 0.060, 0.026, 0.005, 0.001], "edge": [0.001, 0.000, 0.000, 0.000, 0.000]}


def _load_ci():
    """taniteval/taniteval/ci.py, loaded BY FILE (importing the package would pin the tanitad stack).  Search: $E0D_CI_PATH, then the repo layout."""
    cands = [os.environ.get("E0D_CI_PATH", "")]
    for p in HERE.parents:
        cands.append(str(p / "taniteval" / "taniteval" / "ci.py"))
    for c in cands:
        if c and Path(c).is_file():
            spec = importlib.util.spec_from_file_location("e0d_ci", c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["e0d_ci"] = mod
            spec.loader.exec_module(mod)
            mod.__path_sha256__ = hashlib.sha256(Path(c).read_bytes()).hexdigest()
            mod.__path__file = c
            return mod
    raise SystemExit("[e0d] taniteval/taniteval/ci.py not found: set E0D_CI_PATH")


# --------------------------------------------------------------------------------------------------------------------------------------- #
# geometry primitives (numpy; identical semantics to mapsel_lib.dilate: square structuring element, zero padding)
# --------------------------------------------------------------------------------------------------------------------------------------- #
def dilate(mask, r):
    """bool [H, W] -> bool [H, W], square element of half-width r cells (r = 0: unchanged), zero padding outside the raster."""
    m = np.asarray(mask, dtype=bool)
    if r <= 0:
        return m
    a = m.copy()
    for s in range(1, r + 1):
        a[s:] |= m[:-s]
        a[:-s] |= m[s:]
    b = a.copy()
    for s in range(1, r + 1):
        b[:, s:] |= a[:, :-s]
        b[:, :-s] |= a[:, s:]
    return b


def boundary(d):
    """inner boundary of a bool plane: d & dilate(~d, 1), the outermost raster row/column never a boundary cell (H-R9-MAPEDGE-1 definition)."""
    d = np.asarray(d, dtype=bool)
    b = d & dilate(~d, 1)
    b[0, :] = False
    b[-1, :] = False
    b[:, 0] = False
    b[:, -1] = False
    return b


def row_band(h, edges):
    return np.repeat(np.arange(len(edges) - 1), np.diff(edges))[:h]


def band_counts(mask, edges):
    return np.array([np.count_nonzero(mask[a:b]) for a, b in zip(edges[:-1], edges[1:])], np.int64)


# --------------------------------------------------------------------------------------------------------------------------------------- #
# per-window sufficient statistics
# --------------------------------------------------------------------------------------------------------------------------------------- #
def gt_static(gt, edges=EDGES20):
    """Everything that depends on GT alone, computed once per window and shared by every prediction arm."""
    gt = np.asarray(gt)
    nb = len(edges) - 1
    isD = np.isin(gt, DRV_CODES)
    valid = gt != UNSEEN
    J = valid & (gt != NOCLS)
    E = gt == EDGE
    G = gt == LANE
    cls = np.where(isD, 0, np.where(gt == NOCLS, 1, np.where(gt == UNSEEN, 3, 2))).astype(np.int32)
    rb = row_band(gt.shape[0], edges)
    st = {"gt": gt, "isD": isD, "valid": valid, "J": J, "E": E, "G": G, "edges": edges, "rb": rb,
          "kbase": ((rb[:, None].astype(np.int32) * 4 + cls) * 256),
          "gtidx": np.where(gt == UNSEEN, 8, np.minimum(gt, 7)).astype(np.int32),
          "Edil": dilate(E, TOL_EDGE), "nE": band_counts(E, edges),
          "Gd": {r: dilate(G, r) for r in LANE_RS}, "nb": nb}
    gb = isD & dilate(J & ~isD, 1)                                      # GT-BND: GT drivable cells next to a JUDGED non-drivable cell
    gb[0, :] = gb[-1, :] = False
    gb[:, 0] = gb[:, -1] = False
    st["gtbnd"] = _edge_stats(gb, st)
    return st


def _edge_stats(X, st):
    """[nb, 5] = nE, hitE (GT edge cells within 0.3 m of X), nXJ, hitX (judged X cells within 0.3 m of a GT edge), nX."""
    ed = st["edges"]
    if not X.any():
        z = np.zeros(st["nb"], np.int64)
        return np.stack([st["nE"], z, z, z, z], 1)
    Xd = dilate(X, TOL_EDGE)
    XJ = X & st["J"]
    return np.stack([st["nE"], band_counts(st["E"] & Xd, ed), band_counts(XJ, ed), band_counts(XJ & st["Edil"], ed), band_counts(X, ed)], 1)


def plane_stats(pdrv, pedge, code, st, thr_drv=THR_DRV, thr_edge=THR_EDGE, with_edge=True):
    """Per-window statistics of ONE prediction arm.  Shapes (nb = 5 bands):
         drv_hist [nb,4,256]  pdrv-value histogram per GT class (0 drivable-union, 1 nocls, 2 other judged, 3 not seen)
         conf     [nb, 9, 8] (GT value 0..7 | 255 -> row 8) x (decided code)
         lane     [len(LANE_RS), nb, 7]  A=|Pd&Gd&v| B=|Pd&v| C=|Gd&v| tpp=|P&Gd&v| np=|P&v| tpg=|G&Pd&v| ng=|G&v|
         bnd      [len(thr_drv), nb, 5]  boundary of {pdrv >= t}: _edge_stats
         pedge    [len(thr_edge), nb, 5] {pedge >= t} as the edge candidate (only when with_edge)
         code5    [nb, 5]                {code == 5}"""
    nb, ed = st["nb"], st["edges"]
    out = {}
    key = st["kbase"] + np.asarray(pdrv, np.int32)
    out["drv_hist"] = np.bincount(key.ravel(), minlength=nb * 4 * 256).reshape(nb, 4, 256).astype(np.int32)
    assert int(np.asarray(code).max()) <= 7, "decided code above 7"
    key2 = (st["rb"][:, None].astype(np.int32) * 9 + st["gtidx"]) * 8 + np.asarray(code, np.int32)
    out["conf"] = np.bincount(key2.ravel(), minlength=nb * 9 * 8).reshape(nb, 9, 8).astype(np.int32)
    P = np.asarray(code) == LANE
    v = st["valid"]
    lane = np.zeros((len(LANE_RS), nb, 7), np.int64)
    for i, r in enumerate(LANE_RS):
        Pd, Gd, G = dilate(P, r), st["Gd"][r], st["G"]
        lane[i] = np.stack([band_counts(Pd & Gd & v, ed), band_counts(Pd & v, ed), band_counts(Gd & v, ed), band_counts(P & Gd & v, ed),
                            band_counts(P & v, ed), band_counts(G & Pd & v, ed), band_counts(G & v, ed)], 1)
    out["lane"] = lane.astype(np.int32)
    out["bnd"] = np.stack([_edge_stats(boundary(np.asarray(pdrv) >= t), st) for t in thr_drv]).astype(np.int32)
    out["code5"] = _edge_stats(np.asarray(code) == EDGE, st).astype(np.int32)
    if with_edge:
        out["pedge"] = np.stack([_edge_stats(np.asarray(pedge) >= t, st) for t in thr_edge]).astype(np.int32)
    return out


def control_planes(gt):
    """The three GT-only arms.  C0 = constant all-negative map; CP = constant all-positive map (drivable, edge, lane everywhere); GTP = GT as the prediction."""
    z = np.zeros(gt.shape, np.uint8)
    full = np.full(gt.shape, 255, np.uint8)
    gcode = np.where(gt == UNSEEN, 0, gt).astype(np.uint8)
    return {"C0": (z, z, z), "CP": (full, full, np.full(gt.shape, LANE, np.uint8)),
            "GTP": (np.where(np.isin(gt, DRV_CODES), 255, 0).astype(np.uint8), np.where(gt == EDGE, 255, 0).astype(np.uint8), gcode)}


# --------------------------------------------------------------------------------------------------------------------------------------- #
# split / mapping
# --------------------------------------------------------------------------------------------------------------------------------------- #
def sha12_of(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def episode_maps(clip_ids, split):
    """e_index (position of the clip id in the SORTED id list of the battery windows file) -> sha12, and the DEV / FIT / SCORE / HOLD episode lists.
    The index rule was checked against the data (the two clips with no SAM3 GT file are exactly the two episodes whose rasters carry no `gt` plane)."""
    assert list(clip_ids) == sorted(clip_ids), "windows file is not in sorted clip-id order: the e-index rule would not hold"
    sha = [sha12_of(c) for c in clip_ids]
    assert len(set(sha)) == len(sha)
    e_of = {s: i for i, s in enumerate(sha)}
    dev = [e_of[s] for s in split["DEV"]["sha12"]]                   # the split file's own rank order (sha256 ascending)
    hold = [e_of[s] for s in split["HOLD"]["sha12"]]
    assert not set(dev) & set(hold) and len(dev) + len(hold) == len(sha)
    return {"sha": sha, "dev": dev, "fit": dev[0::2], "score": dev[1::2], "hold": hold}


def episodes_for(which, maps):
    if which == "dev":
        return list(maps["dev"])
    if which == "hold" and os.environ.get("E0D_UNSEAL_HOLD") == "yes":
        return list(maps["hold"])
    raise SystemExit("[e0d] HOLD is sealed (REFCV9_DESIGN 5.1): only the Master Mind's sealing step may set E0D_UNSEAL_HOLD=yes")


def assert_disjoint(fit_idx, score_idx, eid, label=""):
    """Thresholds never touch the scored windows: the FIT and SCORE sets share no window and no episode.  Returns the printed n."""
    fi, si = set(map(int, fit_idx)), set(map(int, score_idx))
    if fi & si:
        raise AssertionError(f"[e0d] FIT and SCORE share {len(fi & si)} windows {label}")
    fe, se = {str(eid[i]) for i in fi}, {str(eid[i]) for i in si}
    if fe & se:
        raise AssertionError(f"[e0d] FIT and SCORE share {len(fe & se)} episodes {label}")
    return {"n_fit_windows": len(fi), "n_score_windows": len(si), "n_fit_episodes": len(fe), "n_score_episodes": len(se), "shared_windows": 0, "shared_episodes": 0}


# --------------------------------------------------------------------------------------------------------------------------------------- #
# pass 1: extraction (parallel, detached)
# --------------------------------------------------------------------------------------------------------------------------------------- #
_G = {}


def _init_worker(gt_cache, gt_index):
    _G["cache"] = np.load(gt_cache, mmap_mode="r")
    idx = np.load(gt_index)
    _G["pos"] = {(int(e), int(w)): k for k, (e, w, _h) in enumerate(idx)}


def _load_planes(path):
    z = np.load(path)
    return {k: z[k] for k in ("pdrv", "pedge", "code")}, z


def work(job):
    """Retry wrapper: the dev box is shared and a worker can hit a transient MemoryError (MEASURED 2026-11-03 at 12 workers)."""
    import gc
    for attempt in range(6):
        try:
            return work_once(job)
        except MemoryError:
            gc.collect()
            time.sleep(3 + 3 * attempt)
    return work_once(job)


def work_once(job):
    """job = (e, w, rast_dir, donor_path, check_gt) -> dict of per-arm statistics (flattened 'ARM/key')."""
    e, w, rdir, donor, check = job
    t0 = time.time()
    f = Path(rdir) / f"e{e:03d}_w{w:04d}.npz"
    pl, z = _load_planes(f)
    for k in ("pdrv", "pedge", "code"):
        if pl[k].shape != (H, W) or pl[k].dtype != np.uint8:
            return {"_bad": 1, "e": e, "w": w}
    gt = np.asarray(_G["cache"][_G["pos"][(e, w)]])
    rec = {"e": e, "w": w, "_bad": 0, "pdrv_const": int(pl["pdrv"].min() == pl["pdrv"].max())}
    if check:
        rec["gtcache_checked"] = 1
        rec["gtcache_equal"] = int("gt" in z.files and np.array_equal(z["gt"], gt))
    st = gt_static(gt)
    rec["GT/gtbnd"] = st["gtbnd"].astype(np.int32)
    arms = {"OWN": plane_stats(pl["pdrv"], pl["pedge"], pl["code"], st)}
    if donor:
        dpl, _ = _load_planes(donor)
        arms["DER"] = plane_stats(dpl["pdrv"], dpl["pedge"], dpl["code"], st, with_edge=False)
    for name, (a, b, c) in control_planes(gt).items():
        arms[name] = plane_stats(a, b, c, st, thr_drv=(FIXED_T,), thr_edge=(FIXED_T,))
    for name, d in arms.items():
        for k, v in d.items():
            rec[f"{name}/{k}"] = v
    rec["_s"] = time.time() - t0
    return rec


def cmd_extract(a):
    import multiprocessing as mp

    ci = _load_ci()
    clips = list(json.load(open(a.windows_json, encoding="utf-8")).keys())
    split = json.load(open(a.split_json, encoding="utf-8"))
    maps = episode_maps(clips, split)
    eps = episodes_for(a.split, maps)
    files = {}
    for f in sorted(Path(a.rast).glob("e*_w*.npz")):
        m = NAME.search(f.name)
        files.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    # mapping control: the per-episode banked window count equals the E0i S2 count for every DEV episode
    cnt_ok = all(len(files.get(e, [])) == split["DEV"]["windows_per_episode"][maps["sha"][e]][0] for e in maps["dev"])
    gt_idx = np.load(a.gt_index)
    has = {(int(e), int(w)): int(h) for e, w, h in gt_idx}
    sam3 = {f.split(".")[0] for f in os.listdir(a.sam3_gt) if f.endswith(".sam3mapgt.npz")} if a.sam3_gt else set()
    nogt_eps = sorted({e for e in eps if any(not has[(e, w)] for w in files[e])})
    nogt_all = sorted(e for e in range(len(clips)) if files.get(e) and not any(has[(e, w)] for w in files[e]))
    store_missing = sorted(e for e in range(len(clips)) if maps["sha"][e] not in sam3) if sam3 else None
    # deranged donor: cyclic NEXT DEV episode (sorted by index), position-matched
    ds = sorted(eps)
    donor = {}
    for i, e in enumerate(ds):
        de = ds[(i + 1) % len(ds)]
        for p, w in enumerate(sorted(files[e])):
            donor[(e, w)] = Path(a.rast) / f"e{de:03d}_w{sorted(files[de])[min(p, len(files[de]) - 1)]:04d}.npz"
    jobs, nogt = [], 0
    for e in sorted(eps):
        for w in sorted(files[e]):
            if not has[(e, w)]:
                nogt += 1
                continue
            jobs.append((e, w, a.rast, str(donor[(e, w)]), (len(jobs) % 25 == 0)))
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]
    print(f"[e0d] extract split={a.split} episodes={len(eps)} windows_with_gt={len(jobs)} windows_without_gt={nogt} workers={a.workers}", flush=True)
    t0 = time.time()
    recs = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(a.gt_cache, a.gt_index)) as pool:
        for i, r in enumerate(pool.imap(work, jobs, chunksize=2)):
            recs.append(r)
            if (i + 1) % 100 == 0:
                print(f"[e0d] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    bad = [r for r in recs if r.get("_bad")]
    recs = [r for r in recs if not r.get("_bad")]
    keys = [k for k in recs[0] if "/" in k]
    arr = {k.replace("/", "__"): np.stack([r[k] for r in recs]) for k in keys}
    arr["e"] = np.array([r["e"] for r in recs], np.int64)
    arr["w"] = np.array([r["w"] for r in recs], np.int64)
    arr["pdrv_const"] = np.array([r["pdrv_const"] for r in recs], np.int8)
    arr["gtcache_checked"] = np.array([r.get("gtcache_checked", 0) for r in recs], np.int8)
    arr["gtcache_equal"] = np.array([r.get("gtcache_equal", 0) for r in recs], np.int8)
    meta = {"tool": "e0d_score.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "ci_sha256": ci.__path_sha256__,
            "split": a.split, "n_windows": len(recs), "n_bad": len(bad), "n_windows_without_gt_skipped": nogt, "episodes_without_gt_plane": nogt_eps,
            "dev_fit_episodes": maps["fit"], "dev_score_episodes": maps["score"], "dev_episodes": maps["dev"],
            "dev_hold_split_file_sha256": hashlib.sha256(Path(a.split_json).read_bytes()).hexdigest(),
            "windows_json_sha256": hashlib.sha256(Path(a.windows_json).read_bytes()).hexdigest(),
            "map_check_window_counts_equal_E0i": bool(cnt_ok),
            "map_check_clips_with_no_sam3_gt_file_equal_episodes_without_gt_plane": (None if store_missing is None else store_missing == nogt_all),
            "episodes_without_gt_plane_all139": nogt_all,
            "sam3_gt_missing_e": store_missing, "thr_drv": list(THR_DRV), "thr_edge": list(THR_EDGE), "wall_s": round(time.time() - t0, 1),
            "worker_s_sum": round(sum(r["_s"] for r in recs), 1), "rast": str(a.rast), "gt_cache": str(a.gt_cache)}
    arr["meta_json"] = np.array(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arr)
    print(f"[e0d] wrote {a.out} windows={len(recs)} bad={len(bad)} wall={time.time() - t0:.0f}s ZZE0DEXTRACTDONEZZ", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------------------------- #
# pass 2: thresholds on FIT, score on SCORE
# --------------------------------------------------------------------------------------------------------------------------------------- #
def iou_from(tp, fp, fn):
    d = tp + fp + fn
    return float(tp) / float(d) if d > 0 else float("nan")


def f_from(P, R):
    return 2 * P * R / (P + R) if (P is not None and R is not None and P + R > 0) else 0.0


def fit_drv_threshold(pos, neg):
    """pos / neg [256] pooled pdrv histograms -> (t in 1..255 maximising IoU of {pdrv >= t}, that IoU).  Ties -> the lowest t."""
    cpos = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    cneg = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    tp, fp = cpos[1:], cneg[1:]
    fn = float(pos.sum()) - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), -1.0)
    i = int(np.argmax(iou))
    return i + 1, float(iou[i])


def fit_grid_threshold(stats, thr, bands):
    """stats [T, nb, 5] pooled over the FIT windows -> (threshold with the best pooled F@0.3 m, its F, its P, its R).  Fallback when F is 0 everywhere: FIXED_T-nearest."""
    best = None
    for i, t in enumerate(thr):
        s = stats[i][list(bands)].sum(0)
        R = s[1] / s[0] if s[0] > 0 else None
        P = s[3] / s[2] if s[2] > 0 else None
        F = f_from(P, R)
        if best is None or F > best[1]:
            best = (t, F, P, R)
    if best[1] <= 0:
        t = min(thr, key=lambda x: abs(x - FIXED_T))
        return t, 0.0, None, None
    return best


def _ep_sums(comp, eid):
    ue, inv = np.unique(eid, return_inverse=True)
    S = np.zeros((len(ue), comp.shape[1]))
    np.add.at(S, inv, comp)
    return [S[i] for i in range(len(ue))], [str(x) for x in ue]


def boot(ci, comp, eid, f, n_boot=N_BOOT, n_defined=None):
    """Episode-cluster bootstrap of f(sum of components) -- an additive-count statistic, so resampling per-episode sums IS resampling the windows of
    the drawn episodes.  comp [N, k] per-window components."""
    comp = np.asarray(comp, np.float64)
    point = f(comp.sum(0))
    if len(comp) == 0 or not np.isfinite(point):
        return {"point": None, "n_windows": int(len(comp)), "n_episodes": int(len(set(map(str, eid)))), "undefined": True}
    items, names = _ep_sums(comp, eid)
    r = ci.stat_cluster_bootstrap(items, names, lambda L: f(np.sum(L, 0)), n_boot=n_boot, seed=0)
    r["n_windows"] = int(len(comp) if n_defined is None else n_defined)
    r["n_episodes"] = int(len(names))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}


def paired(ci, comp_a, comp_b, eid, f, n_boot=N_BOOT, n_defined=None):
    A, names = _ep_sums(np.asarray(comp_a, np.float64), eid)
    B, _ = _ep_sums(np.asarray(comp_b, np.float64), eid)
    r = ci.paired_stat_cluster_bootstrap(A, B, names, lambda L: f(np.sum(L, 0)), n_boot=n_boot, seed=0)
    r["n_windows"] = int(len(comp_a) if n_defined is None else n_defined)
    r["n_episodes"] = int(len(names))
    return r


class Bank:
    """The extracted per-window statistics with FIT / SCORE selection.  `Bank(path)` or `Bank(z=<dict of arrays>)` (the tests)."""

    def __init__(self, path=None, z=None, quiet=False):
        if z is None:
            zz = np.load(path, allow_pickle=False)
            z = {k: zz[k] for k in zz.files}
        self.z = z
        self.meta = json.loads(str(z["meta_json"]))
        self.e, self.w = z["e"], z["w"]
        self.eid = np.array([f"e{int(x):03d}" for x in self.e])
        fit_e, score_e = set(self.meta["dev_fit_episodes"]), set(self.meta["dev_score_episodes"])
        self.fit = np.array([i for i, x in enumerate(self.e) if int(x) in fit_e], dtype=np.int64)
        self.score = np.array([i for i, x in enumerate(self.e) if int(x) in score_e], dtype=np.int64)
        self.all = np.arange(len(self.e))
        self.disjoint = assert_disjoint(self.fit, self.score, self.eid)
        if not quiet:
            d = self.disjoint
            print(f"[e0d] FIT n={d['n_fit_windows']} windows / {d['n_fit_episodes']} episodes | SCORE n={d['n_score_windows']} windows / "
                  f"{d['n_score_episodes']} episodes | shared windows 0, shared episodes 0", flush=True)

    def g(self, arm, key):
        return self.z[f"{arm}__{key}"]

    def has(self, arm, key):
        return f"{arm}__{key}" in self.z


# ----- count extraction (all return [n, k] float64 per-window components) ---------------------------------------------------------------- #
def drv_counts(bank, arm, idx, bands, t, variant="valid"):
    h = bank.g(arm, "drv_hist")[idx][:, list(bands)].sum(1).astype(np.float64)          # [n, 4, 256]
    pos = h[:, 0]
    neg = h[:, 2] if variant == "J" else h[:, 1] + h[:, 2]
    return np.stack([pos[:, t:].sum(1), neg[:, t:].sum(1), pos[:, :t].sum(1)], 1)          # tp, fp, fn


def drv_pooled_hist(bank, arm, idx, bands, variant="valid"):
    h = bank.g(arm, "drv_hist")[idx][:, list(bands)].sum((0, 1)).astype(np.float64)
    return h[0], (h[2] if variant == "J" else h[1] + h[2])


def lane_counts(bank, arm, idx, bands, r):
    return bank.g(arm, "lane")[idx][:, LANE_RS.index(r)][:, list(bands)].sum(1).astype(np.float64)    # [n, 7]


def edge_rows(bank, arm, key, ti, idx, bands):
    a = bank.g(arm, key)
    return (a[idx][:, ti] if a.ndim == 4 else a[idx])[:, list(bands)].sum(1).astype(np.float64)       # [n, 5] nE hitE nXJ hitX nX


def iou_from(tp, fp, fn):
    d = tp + fp + fn
    return float(tp) / float(d) if d > 0 else float("nan")


def f_from(P, R):
    return 2 * P * R / (P + R) if (P is not None and R is not None and P + R > 0) else 0.0


def f_iou(s):
    return iou_from(s[0], s[1], s[2])


def f_ioudil(s):
    return iou_from(s[0], s[1] - s[0], s[2] - s[0])                  # A, B, C -> A / (B + C - A)


def f_tol(s):
    """the project's boundary-tolerant IoU: P_k = tpp/np, R_k = tpg/ng, F = 2PR/(P+R), IoU = F/(2-F)  (map_head_hires.tolerant_iou_from_counts)."""
    if s[4] <= 0 or s[6] <= 0:
        return float("nan")
    F = f_from(s[3] / s[4], s[5] / s[6])
    return F / (2.0 - F)


def f_ratio(s):
    return float(s[0] / s[1]) if s[1] > 0 else float("nan")


def recall_comp(s):
    """per-window recall, defined when >= MIN_CELLS GT edge cells: components [recall, 1] ([0, 0] when undefined); sum/sum = the pooled window mean."""
    ok = s[:, 0] >= MIN_CELLS
    return np.stack([np.where(ok, s[:, 1] / np.maximum(s[:, 0], 1), 0.0), ok.astype(np.float64)], 1)


def prec_comp(s):
    ok = s[:, 2] >= MIN_CELLS
    return np.stack([np.where(ok, s[:, 3] / np.maximum(s[:, 2], 1), 0.0), ok.astype(np.float64)], 1)


def pooled_rec_comp(s):
    return np.stack([s[:, 1], s[:, 0]], 1)


# ----- threshold fitting ---------------------------------------------------------------------------------------------------------------- #
def fit_drv_threshold(pos, neg):
    """pos / neg [256] pooled pdrv histograms -> (t in 1..255 maximising the IoU of {pdrv >= t}, that IoU).  Ties -> the lowest t."""
    cpos = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    cneg = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    tp, fp = cpos[1:], cneg[1:]
    fn = float(pos.sum()) - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), -1.0)
    i = int(np.argmax(iou))
    return i + 1, float(iou[i])


def fit_grid_threshold(stats, thr, bands):
    """stats [T, nb, 5] pooled over the FIT windows -> (threshold with the best pooled F@0.3 m, F, P, R).  F == 0 everywhere -> the grid point nearest 128."""
    best = None
    for i, t in enumerate(thr):
        s = stats[i][list(bands)].sum(0)
        R = s[1] / s[0] if s[0] > 0 else None
        P = s[3] / s[2] if s[2] > 0 else None
        F = f_from(P, R)
        if best is None or F > best[1]:
            best = (t, F, P, R)
    if best[1] <= 0:
        return min(thr, key=lambda x: abs(x - FIXED_T)), 0.0, None, None
    return best


def thresholds_from(bank, idx, arm="OWN"):
    """Per-class thresholds fitted on the windows `idx` ONLY (never on others).  perband = one per class per 4-band; global = one per class over all bands."""
    out = {"perband": {"drv": {}, "bnd": {}, "pedge": {}}, "global": {}, "diag": {}, "fit_windows": int(len(idx)), "fit_episodes": int(len(set(bank.eid[idx])))}
    for name, bands in list(BAND4.items()) + [("all", (0, 1, 2, 3, 4))]:
        pos, neg = drv_pooled_hist(bank, arm, idx, bands)
        t, iou = fit_drv_threshold(pos, neg)
        bt = fit_grid_threshold(bank.g(arm, "bnd")[idx].sum(0), THR_DRV, bands)
        et = fit_grid_threshold(bank.g(arm, "pedge")[idx].sum(0), THR_EDGE, bands)
        if name == "all":
            out["global"] = {"drv": int(t), "bnd": int(bt[0]), "pedge": int(et[0])}
            out["diag"]["global"] = {"drv_iou_fit": round(iou, 5), "bnd_F_fit": round(bt[1], 5), "pedge_F_fit": round(et[1], 5)}
        else:
            out["perband"]["drv"][name], out["perband"]["bnd"][name], out["perband"]["pedge"][name] = int(t), int(bt[0]), int(et[0])
            out["diag"][name] = {"drv_iou_fit": round(iou, 5), "bnd_F_fit": round(bt[1], 5), "bnd_P_fit": None if bt[2] is None else round(bt[2], 5),
                                 "bnd_R_fit": None if bt[3] is None else round(bt[3], 5), "pedge_F_fit": round(et[1], 5)}
    return out


# ----- bootstrap wrappers (taniteval/ci.py) ----------------------------------------------------------------------------------------------- #
def _ep_sums(comp, eid):
    ue, inv = np.unique(eid, return_inverse=True)
    S = np.zeros((len(ue), comp.shape[1]))
    np.add.at(S, inv, comp)
    return [S[i] for i in range(len(ue))], [str(x) for x in ue]


def boot(ci, comp, eid, f, n_boot=N_BOOT, n_defined=None):
    """Episode-cluster bootstrap of f(sum of components).  The statistics are additive counts, so resampling per-episode SUMS is exactly resampling the
    windows of the drawn episodes.  Point = the full-set value."""
    comp = np.asarray(comp, np.float64)
    eid = np.asarray(eid)
    point = f(comp.sum(0)) if len(comp) else float("nan")
    if not np.isfinite(point):
        return {"point": None, "n_windows": int(len(comp)), "n_episodes": int(len(set(map(str, eid)))), "undefined": True}
    items, names = _ep_sums(comp, eid)
    r = ci.stat_cluster_bootstrap(items, names, lambda L: f(np.sum(L, 0)), n_boot=n_boot, seed=0)
    r["n_windows"] = int(len(comp) if n_defined is None else n_defined)
    r["n_episodes"] = int(len(names))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}


def paired(ci, comp_a, comp_b, eid, f, n_boot=N_BOOT, n_defined=None):
    eid = np.asarray(eid)
    A, names = _ep_sums(np.asarray(comp_a, np.float64), eid)
    B, _ = _ep_sums(np.asarray(comp_b, np.float64), eid)
    r = ci.paired_stat_cluster_bootstrap(A, B, names, lambda L: f(np.sum(L, 0)), n_boot=n_boot, seed=0)
    r["n_windows"] = int(len(comp_a) if n_defined is None else n_defined)
    r["n_episodes"] = int(len(names))
    return r


def cat(parts):
    return np.concatenate(parts, 0)


# ----- the MAP-3 / MAP-4 rows ----------------------------------------------------------------------------------------------------------- #
def _t(thr, kind, name, tag):
    return int(thr["perband"][kind][name] if tag == "perband" else thr["global"][kind])


def band_components(bank, arm, folds, name, bands, tag):
    """Per-window components of every headline metric for ONE band4, over the (idx, thr) folds -- thresholds differ by fold (cross-fit) but never touch their own windows."""
    c = {k: [] for k in ("drv", "drvJ", "drv128", "ld2", "l0", "tol2", "tol5", "br", "bp", "bpool", "br128", "er", "ep", "c5")}
    eid = []
    for idx, thr in folds:
        td, tb, te = _t(thr, "drv", name, tag), _t(thr, "bnd", name, tag), _t(thr, "pedge", name, tag)
        c["drv"].append(drv_counts(bank, arm, idx, bands, td))
        c["drvJ"].append(drv_counts(bank, arm, idx, bands, td, "J"))
        c["drv128"].append(drv_counts(bank, arm, idx, bands, FIXED_T))
        c["ld2"].append(lane_counts(bank, arm, idx, bands, 2)[:, :3])
        c["l0"].append(lane_counts(bank, arm, idx, bands, 0)[:, :3])
        c["tol2"].append(lane_counts(bank, arm, idx, bands, 2))
        c["tol5"].append(lane_counts(bank, arm, idx, bands, 5))
        s = edge_rows(bank, arm, "bnd", THR_DRV.index(tb), idx, bands)
        c["br"].append(recall_comp(s))
        c["bp"].append(prec_comp(s))
        c["bpool"].append(pooled_rec_comp(s))
        c["br128"].append(recall_comp(edge_rows(bank, arm, "bnd", THR_DRV.index(FIXED_T), idx, bands)))
        se = edge_rows(bank, arm, "pedge", THR_EDGE.index(te), idx, bands)
        c["er"].append(recall_comp(se))
        c["ep"].append(prec_comp(se))
        c["c5"].append(recall_comp(edge_rows(bank, arm, "code5", 0, idx, bands)))
        eid.append(bank.eid[idx])
    return {k: cat(v) for k, v in c.items()}, np.concatenate(eid)


def score_rows(ci, bank, arm, folds, tag, n_boot):
    rows = {}
    for name, bands in BAND4.items():
        c, eid = band_components(bank, arm, folds, name, bands, tag)
        thr0 = folds[0][1]
        r = {"threshold": {"drv": _t(thr0, "drv", name, tag), "bnd": _t(thr0, "bnd", name, tag), "pedge": _t(thr0, "pedge", name, tag)}}
        if len(folds) > 1:
            r["threshold_by_fold"] = [{"drv": _t(t, "drv", name, tag), "bnd": _t(t, "bnd", name, tag), "pedge": _t(t, "pedge", name, tag)} for _, t in folds]
        d = lambda m: int(m.sum())  # noqa: E731
        r["drv_iou"] = boot(ci, c["drv"], eid, f_iou, n_boot, n_defined=int(((c["drv"][:, 0] + c["drv"][:, 2]) > 0).sum()))
        r["drv_iou_J_nocls_ignored"] = boot(ci, c["drvJ"], eid, f_iou, n_boot)
        r["drv_iou_t128"] = boot(ci, c["drv128"], eid, f_iou, n_boot)
        r["lane_ioudil_0p5m"] = boot(ci, c["ld2"], eid, f_ioudil, n_boot, n_defined=int((c["ld2"][:, 2] > 0).sum()))
        r["lane_iou_exact_10cm"] = boot(ci, c["l0"], eid, f_ioudil, n_boot, n_defined=int((c["l0"][:, 2] > 0).sum()))
        r["lane_tolF_k2_0p2m"] = boot(ci, c["tol2"], eid, f_tol, n_boot, n_defined=int((c["tol2"][:, 6] > 0).sum()))
        r["lane_tolF_k5_0p5m"] = boot(ci, c["tol5"], eid, f_tol, n_boot, n_defined=int((c["tol5"][:, 6] > 0).sum()))
        r["bnd_recall_0p3m"] = boot(ci, c["br"], eid, f_ratio, n_boot, n_defined=d(c["br"][:, 1]))
        r["bnd_precision_0p3m"] = boot(ci, c["bp"], eid, f_ratio, n_boot, n_defined=d(c["bp"][:, 1]))
        r["bnd_recall_pooled_cells"] = boot(ci, c["bpool"], eid, f_ratio, n_boot)
        r["bnd_recall_t128"] = boot(ci, c["br128"], eid, f_ratio, n_boot, n_defined=d(c["br128"][:, 1]))
        r["edgeclass_recall_0p3m"] = boot(ci, c["er"], eid, f_ratio, n_boot, n_defined=d(c["er"][:, 1]))
        r["edgeclass_precision_0p3m"] = boot(ci, c["ep"], eid, f_ratio, n_boot, n_defined=d(c["ep"][:, 1]))
        r["code5_recall_0p3m"] = boot(ci, c["c5"], eid, f_ratio, n_boot, n_defined=d(c["c5"][:, 1]))
        rows[name] = r
    return rows


def control_arm_points(bank, arm, idx, bands):
    """Point readings of a control arm built from BINARY planes (C0 / CP / GTP): its threshold is irrelevant (any t in 1..255 reads the same); the single bnd / pedge column is index 0."""
    h = bank.g(arm, "drv_hist")[idx][:, list(bands)].sum(1).astype(np.float64)
    pos, neg = h[:, 0], h[:, 1] + h[:, 2]
    tp, fp, fn = pos[:, FIXED_T:].sum(), neg[:, FIXED_T:].sum(), pos[:, :FIXED_T].sum()
    lc0, lc2, lc5 = (lane_counts(bank, arm, idx, bands, r) for r in LANE_RS)
    se = edge_rows(bank, arm, "pedge", 0, idx, bands)
    sb = edge_rows(bank, arm, "bnd", 0, idx, bands)
    return {"drv_iou": iou_from(tp, fp, fn), "lane_ioudil_0p5m": f_ioudil(lc2[:, :3].sum(0)), "lane_iou_exact_10cm": f_ioudil(lc0[:, :3].sum(0)),
            "lane_tolF_k2": f_tol(lc2.sum(0)), "lane_tolF_k5": f_tol(lc5.sum(0)),
            "bnd_recall_0p3m": f_ratio(recall_comp(sb).sum(0)), "edgeclass_recall_0p3m": f_ratio(recall_comp(se).sum(0)),
            "edgeclass_precision_0p3m": f_ratio(prec_comp(se).sum(0)), "code5_recall_0p3m": f_ratio(recall_comp(edge_rows(bank, arm, "code5", 0, idx, bands)).sum(0))}


def independent_prevalence(bank, idx, bands):
    """Analytic no-information values from OWN's class histograms (a different code path from the CP arm): drivable prevalence, dilated-lane prevalence."""
    h = bank.g("OWN", "drv_hist")[idx][:, list(bands)].sum((0, 1)).astype(np.float64)
    valid = h[0].sum() + h[1].sum() + h[2].sum()
    lane_c = lane_counts(bank, "OWN", idx, bands, 2)[:, 2].sum()                      # |Gd & valid|, a GT-only count
    return {"drv_iou_allpos": float(h[0].sum() / valid), "lane_ioudil_allpos": float(lane_c / valid)}


# ----- reproductions / cross-checks ------------------------------------------------------------------------------------------------------- #
def mapedge_protocol(ci, bank, idx, n_boot):
    """H-R9-MAPEDGE-1 under ITS OWN protocol on the given windows: region rows 0..599, windows with >= 20 GT edge cells (precision: >= 20 X&J cells), OWN-BND = boundary of
    pdrv >= 128, tolerance 3 cells, pooled window mean, episode-cluster bootstrap.  Arms: OWN-BND, OWN-CODE5, DER-BND, GT-BND."""
    eid = bank.eid[idx]
    t128 = THR_DRV.index(FIXED_T)
    arms = {"OWN-BND": bank.g("OWN", "bnd")[idx][:, t128], "OWN-CODE5": bank.g("OWN", "code5")[idx], "DER-BND": bank.g("DER", "bnd")[idx][:, t128],
            "GT-BND": bank.z["GT__gtbnd"][idx]}
    res = {}
    for nm, a in arms.items():
        s = a[:, :3].sum(1).astype(np.float64)
        rc, pc = recall_comp(s), prec_comp(s)
        res[nm] = {"rec": boot(ci, rc, eid, f_ratio, n_boot, n_defined=int(rc[:, 1].sum())), "prec": boot(ci, pc, eid, f_ratio, n_boot, n_defined=int(pc[:, 1].sum()))}
        for bn, bi in (("0-20m", 0), ("20-40m", 1), ("40-60m", 2)):
            rcb = recall_comp(a[:, bi].astype(np.float64))
            res[nm][f"rec_{bn}"] = boot(ci, rcb, eid, f_ratio, n_boot, n_defined=int(rcb[:, 1].sum()))
    own = recall_comp(arms["OWN-BND"][:, :3].sum(1).astype(np.float64))
    for nm in ("OWN-CODE5", "DER-BND"):
        oth = recall_comp(arms[nm][:, :3].sum(1).astype(np.float64))
        both = ((own[:, 1] > 0) & (oth[:, 1] > 0)).astype(float)
        res[f"B_rec_OWN-BND_minus_{nm}"] = paired(ci, np.stack([own[:, 0] * both, both], 1), np.stack([oth[:, 0] * both, both], 1), eid, f_ratio, n_boot, n_defined=int(both.sum()))
    return res


def decided_class_iou(ci, bank, idx, n_boot):
    """perc8's decided-class IoU (pooled over the windows, per 20-m band, valid = GT != 255) from the code-vs-GT confusion counts -- a convention cross-check."""
    eid = bank.eid[idx]
    conf = bank.g("OWN", "conf")[idx].astype(np.float64)           # [n, nb, 9, 8]
    out = {}
    for cname, c in (("drivable", 1), ("lane", 2), ("edge", 5)):
        out[cname] = {}
        for bn, bb in BAND5.items():
            cb = conf[:, list(bb)].sum(1)                            # [n, 9, 8]: GT value (8 = not seen) x decided code
            tp = cb[:, c, c]
            fn = cb[:, c, :].sum(1) - tp
            fp = cb[:, :8, c].sum(1) - tp
            out[cname][bn] = boot(ci, np.stack([tp, fp, fn], 1), eid, f_iou, n_boot)
    return out


def inside(pub, cell):
    return bool(cell and cell.get("lo") is not None and cell["lo"] <= pub <= cell["hi"])


# ----- pass 2 driver ----------------------------------------------------------------------------------------------------------------------- #
def cmd_read(a):
    ci = _load_ci()
    t0 = time.time()
    bank = Bank(a.stats)
    nb = a.n_boot
    thr_fit = thresholds_from(bank, bank.fit)
    thr_swap = thresholds_from(bank, bank.score)
    thr_dev = thresholds_from(bank, bank.all)
    print(f"[e0d] thresholds fitted (FIT only): {json.dumps(thr_fit['perband'])}", flush=True)
    res = {"tool": "e0d_score.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "ci_sha256": ci.__path_sha256__, "extract_meta": bank.meta,
           "protocol": "see e0d_score.py docstring and E0d_CARD.md",
           "estimator": "episode-cluster bootstrap (taniteval/ci.py stat_cluster_bootstrap / paired_stat_cluster_bootstrap), B %d, seed 0, over EPISODES -- 'another draw of EPISODES' "
                        "only; ONE training seed (refcv8); the map head is deterministic given the frames. IoU = pooled counts; recall = pooled window mean (windows with >= 20 GT edge cells)." % nb,
           "tier": "perception instrument, map head vs SAM3 GT raster, label level (T0); not a planning or driving metric",
           "disjointness": bank.disjoint,
           "thresholds_fit_on_FIT": thr_fit, "thresholds_fit_on_SCORE_swapped_fold": thr_swap, "thresholds_fit_on_all_DEV_for_a_later_HOLD_read": thr_dev}
    res["PRIMARY_SCORE_half_thresholds_from_FIT_perband"] = score_rows(ci, bank, "OWN", [(bank.score, thr_fit)], "perband", nb)
    print(f"[e0d] primary done {time.time() - t0:.0f}s", flush=True)
    res["BESIDE_FIT_half_thresholds_from_SCORE_swapped"] = score_rows(ci, bank, "OWN", [(bank.fit, thr_swap)], "perband", nb)
    res["BESIDE_CROSSFIT_all_DEV_each_window_scored_with_thresholds_from_the_other_half"] = score_rows(ci, bank, "OWN", [(bank.score, thr_fit), (bank.fit, thr_swap)], "perband", nb)
    res["BESIDE_SCORE_half_one_global_threshold_per_class_from_FIT"] = score_rows(ci, bank, "OWN", [(bank.score, thr_fit)], "global", nb)
    print(f"[e0d] rows done {time.time() - t0:.0f}s", flush=True)
    # ---- controls
    ctl = {}
    sc = bank.score
    pts = {arm: {nm: control_arm_points(bank, arm, sc, bd) for nm, bd in BAND4.items()} for arm in ("C0", "CP", "GTP")}
    prev = {nm: independent_prevalence(bank, sc, bd) for nm, bd in BAND4.items()}
    zero_keys = ("drv_iou", "lane_ioudil_0p5m", "lane_iou_exact_10cm", "bnd_recall_0p3m", "edgeclass_recall_0p3m", "code5_recall_0p3m")
    ctl["K_const0_reads_zero"] = {"pass": all(pts["C0"][nm][k] == 0.0 or pts["C0"][nm][k] != pts["C0"][nm][k] for nm in BAND4 for k in zero_keys), "values_0-20": pts["C0"]["0-20"]}
    ctl["K_constP_reads_prevalence"] = {
        "pass": all(abs(pts["CP"][nm]["drv_iou"] - prev[nm]["drv_iou_allpos"]) < 1e-12 and abs(pts["CP"][nm]["lane_ioudil_0p5m"] - prev[nm]["lane_ioudil_allpos"]) < 1e-12
                    and pts["CP"][nm]["bnd_recall_0p3m"] == 0.0 and abs(pts["CP"][nm]["edgeclass_recall_0p3m"] - 1.0) < 1e-12 for nm in BAND4),
        "note": "all-positive map: IoU = class prevalence (independent calc from OWN's class histograms); a constant map has NO boundary, so boundary recall is exactly 0; "
                "an all-positive edge candidate has recall 1 and precision = the edge-neighbourhood prevalence",
        "values_0-20": pts["CP"]["0-20"], "independent_prevalence_0-20": prev["0-20"]}
    one_keys = ("drv_iou", "lane_ioudil_0p5m", "lane_iou_exact_10cm", "lane_tolF_k2", "lane_tolF_k5", "edgeclass_recall_0p3m", "edgeclass_precision_0p3m", "code5_recall_0p3m")
    ctl["K_GT_as_prediction_reads_one"] = {"pass": all(abs(pts["GTP"][nm][k] - 1.0) < 1e-12 for nm in BAND4 for k in one_keys), "values_0-20": pts["GTP"]["0-20"],
                                           "GTP_drivable_boundary_recall_per_band (not 1.0 by construction)": {nm: pts["GTP"][nm]["bnd_recall_0p3m"] for nm in BAND4}}
    co0, eo0 = band_components(bank, "OWN", [(sc, thr_fit)], "0-20", BAND4["0-20"], "perband")
    idp = paired(ci, co0["br"], co0["br"], eo0, f_ratio, nb)
    ctl["K_identity_paired_zero"] = {"pass": bool(idp["delta"] == 0 and idp["lo"] == 0 and idp["hi"] == 0), "cell": idp}
    der = {}
    for nm in ("0-20", "20-40"):
        bd = BAND4[nm]
        co, eo = band_components(bank, "OWN", [(sc, thr_fit)], nm, bd, "perband")
        td, tb = _t(thr_fit, "drv", nm, "perband"), _t(thr_fit, "bnd", nm, "perband")
        der[f"drv_iou_{nm}"] = paired(ci, co["drv"], drv_counts(bank, "DER", sc, bd, td), eo, f_iou, nb)
        der[f"lane_ioudil_{nm}"] = paired(ci, co["ld2"], lane_counts(bank, "DER", sc, bd, 2)[:, :3], eo, f_ioudil, nb)
        ds = recall_comp(edge_rows(bank, "DER", "bnd", THR_DRV.index(tb), sc, bd))
        both = ((co["br"][:, 1] > 0) & (ds[:, 1] > 0)).astype(float)
        der[f"bnd_recall_{nm}"] = paired(ci, np.stack([co["br"][:, 0] * both, both], 1), np.stack([ds[:, 0] * both, both], 1), eo, f_ratio, nb, n_defined=int(both.sum()))
    ctl["K_deranged_donor_lower_than_own"] = {"pass": all(v["separated"] and v["delta"] > 0 for v in der.values()), "cells": der,
                                              "note": "FLAG only, not a VOID condition: a row that is not separated means window-specificity is unproven for that row"}
    pdc = bank.z["pdrv_const"]
    ctl["K_decode"] = {"pass": bool(int(pdc.sum()) <= 0.05 * len(pdc)), "n_windows": int(len(pdc)), "n_constant_pdrv": int(pdc.sum())}
    chk = bank.z["gtcache_checked"].astype(bool)
    ctl["K_gt_cache_equals_banked_gt"] = {"pass": bool(chk.sum() > 0 and bank.z["gtcache_equal"][chk].all()), "n_checked": int(chk.sum()), "n_equal": int(bank.z["gtcache_equal"][chk].sum())}
    ctl["K_disjoint_fit_score"] = {"pass": True, **bank.disjoint}
    ctl["K_mapping_e_index_to_clip"] = {"pass": bool(bank.meta["map_check_window_counts_equal_E0i"] and bank.meta["map_check_clips_with_no_sam3_gt_file_equal_episodes_without_gt_plane"] in (True, None)),
                                        "window_counts_equal_E0i_S2_counts": bank.meta["map_check_window_counts_equal_E0i"],
                                        "clips_without_sam3_gt_file_equal_episodes_without_gt_plane": bank.meta["map_check_clips_with_no_sam3_gt_file_equal_episodes_without_gt_plane"]}
    res["controls"] = ctl
    me = mapedge_protocol(ci, bank, bank.all, nb)
    res["REPRO_H-R9-MAPEDGE-1_own_protocol_on_DEV"] = {
        "published_all_eval139_INHERITED": PUB_MAPEDGE, "dev_windows": me,
        "published_pooled_recall_inside_DEV_CI": inside(PUB_MAPEDGE["rec"][0], me["OWN-BND"]["rec"]),
        "published_0-20m_recall_inside_DEV_CI": inside(PUB_MAPEDGE["rec_0-20m"][0], me["OWN-BND"]["rec_0-20m"]),
        "published_GT-BND_ceiling_inside_DEV_CI": inside(PUB_MAPEDGE["gt_bnd_rec"], me["GT-BND"]["rec"]),
        "note": "DEV is ~half of eval139 (70 of 139 episodes): a consistency check, not an equality; the exact all-windows reproduction needs the HOLD rasters, which stay unread."}
    dc = decided_class_iou(ci, bank, bank.all, nb)
    res["XCHECK_perc8_decided_class_IoU_on_DEV"] = {"published_perc8_all_eval139_INHERITED": PUB_PERC8, "dev_windows": dc,
                                                     "published_inside_DEV_CI": {c: [inside(PUB_PERC8[c][i], dc[c][bn]) for i, bn in enumerate(BAND5)] for c in PUB_PERC8}}
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"] and k != "K_deranged_donor_lower_than_own")
    res["verdict"] = "VOID (control failed: %s)" % ctl_fail if ctl_fail else "BASELINE-READ (hard controls pass)"
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "E0D_RESULT.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    write_table(res, out / "E0D_TABLE.md")
    print(f"[e0d] {res['verdict']} wall={res['wall_s']}s ZZE0DREADDONEZZ", flush=True)
    return 0 if not ctl_fail else 1


def _c(cell, dp=3):
    if not cell or cell.get("point") is None:
        return "n/a"
    return f"{cell['point']:.{dp}f} [{cell['lo']:.{dp}f}, {cell['hi']:.{dp}f}] (n_win {cell['n_windows']}, n_ep {cell['n_episodes']})"


def write_table(res, path):
    L = ["# E0d result table (generated by e0d_score.py; numbers are the raw/E0D_RESULT.json cells)", "", "verdict: " + res["verdict"], "", "estimator: " + res["estimator"], "tier: " + res["tier"], ""]
    for sec, ttl in (("PRIMARY_SCORE_half_thresholds_from_FIT_perband", "PRIMARY -- SCORE half, per-class per-band thresholds fitted on FIT"),
                     ("BESIDE_CROSSFIT_all_DEV_each_window_scored_with_thresholds_from_the_other_half", "BESIDE -- 2-fold cross-fit over all DEV (every window scored with thresholds fitted on the other half)")):
        L += [f"## {ttl}", "", "| band | drivable-union IoU (MAP-4) | lane tol-IoU 0.5 m (L-DIL, MAP-3a) | lane exact 10 cm IoU | boundary recall @0.3 m (MAP-3b) | boundary precision @0.3 m | edge-class recall |",
              "|---|---|---|---|---|---|---|"]
        for bn, r in res[sec].items():
            L.append(f"| {bn} m | {_c(r['drv_iou'])} | {_c(r['lane_ioudil_0p5m'])} | {_c(r['lane_iou_exact_10cm'])} | {_c(r['bnd_recall_0p3m'])} | {_c(r['bnd_precision_0p3m'])} | {_c(r['edgeclass_recall_0p3m'])} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------------------------------------------------------- #
# analytic worlds (no data)
# --------------------------------------------------------------------------------------------------------------------------------------- #
def road_world(h=100, w=60, half=15, edges=(0, 20, 40, 60, 80, 100)):
    """Straight road of 2*half columns centred on w//2; GT: drivable 1 inside, edge 5 on the two first non-drivable columns, sidewalk 7 beyond."""
    c = w // 2
    cols = np.arange(w)
    inside_ = np.abs(cols - c + 0.5) <= half
    gt = np.where(inside_, 1, 7).astype(np.uint8)[None, :].repeat(h, 0)
    gt[:, [c - half - 1, c + half]] = EDGE
    pdrv = np.where(inside_, 255, 0).astype(np.uint8)[None, :].repeat(h, 0)
    return gt, pdrv, edges


def selftest():
    out = {}
    gt, pdrv, edges = road_world()
    st = gt_static(gt, edges)
    z = np.zeros_like(gt)
    s = plane_stats(pdrv, z, z, st, thr_drv=(128,), thr_edge=(128,))["bnd"][0].sum(0).astype(float)
    out["K1_analytic_road"] = {"pass": bool(s[1] == s[0] and s[3] == s[2] and s[0] > 0), "recall": s[1] / s[0], "precision": s[3] / s[2]}
    s2 = plane_stats(np.roll(pdrv, 10, axis=1), z, z, st, thr_drv=(128,), thr_edge=(128,))["bnd"][0].sum(0).astype(float)      # 1.0 m lateral shift against the GT
    out["K2_shifted_road"] = {"pass": bool(s2[1] == 0), "recall": s2[1] / s2[0]}
    g2 = np.zeros((100, 60), np.uint8)
    g2[:, 30] = LANE
    st2 = gt_static(g2, edges)
    width = dilate(g2 == LANE, LANE_PRIMARY_R).sum(1)
    out["K_dil_line_width"] = {"pass": bool((width == 5).all()), "cells_per_row": int(width[0])}
    vals = {}
    for off in (0, 1, 4, 5):
        p = np.zeros_like(g2)
        p[:, 30 + off] = LANE
        L = plane_stats(np.zeros_like(g2), np.zeros_like(g2), p, st2, thr_drv=(128,), thr_edge=(128,))["lane"][LANE_RS.index(2)].sum(0).astype(float)
        vals[off] = f_ioudil(L[:3])
    out["K_lane_ioudil_known_offsets"] = {"pass": bool(vals[0] == 1.0 and abs(vals[1] - 4 / 6) < 1e-12 and abs(vals[4] - 1 / 9) < 1e-12 and vals[5] == 0.0), "values": vals,
                                          "expected": {"0": 1.0, "1": 4 / 6, "4": 1 / 9, "5": 0.0}}
    return out


# --------------------------------------------------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    e = sp.add_parser("extract")
    e.add_argument("--rast", default="F:/refcv9_mapsel/rast_s0")
    e.add_argument("--gt-cache", default="F:/r9e0/e0h/gt_cache.npy")
    e.add_argument("--gt-index", default="F:/r9e0/e0h/gt_cache_index.npy")
    e.add_argument("--windows-json", default="C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/windows_s0.json")
    e.add_argument("--split-json", required=True)
    e.add_argument("--sam3-gt", default="E:/refcv6_eval_kit/data/sam3_gt_v3_eval")
    e.add_argument("--split", default="dev", choices=("dev", "hold"))
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=10)
    e.add_argument("--limit", type=int, default=0, help="smoke: spread this many windows over DEV")
    r = sp.add_parser("read")
    r.add_argument("--stats", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.cmd == "selftest":
        res = selftest()
        print(json.dumps(res, indent=1))
        return 0 if all(v["pass"] for v in res.values()) else 1
    return cmd_extract(a) if a.cmd == "extract" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
