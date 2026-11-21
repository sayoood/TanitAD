"""e0h_dose.py -- E0h (REFCV9_DESIGN.md s4.1): map-quality dose-response of the planner's sidewalk rate.  CPU, 0 GPU.

Question: how PRECISE must the planner's map be?  MAPPATH (PREREG_MAPPATH.md + amendments 1-2) found refcv8's OWN map converts nothing (cross-fitted weight 0 in 5/5 folds) while the
TRUE map does (GT-T1 2.19 % vs BASE 17.92 % sidewalk).  Here the planner's map is the GT raster plus CALIBRATED FALSE sidewalk flags, and MAPPATH's cross-fitted soft cost
(`mapsoft_path.py`, imported UNCHANGED from the banked package) is re-run on MAPSEL's banked progress-matched fan; the readout is the planner's S6c sidewalk rate.

Map model (frozen in E0h_CARD.md): flags = every GT sidewalk cell (recall 1 everywhere) + square FALSE stamps (`--stamp-m` metres, default 1.0) placed uniformly at random on SEEN,
non-sidewalk cells INSIDE the MAP-2 corridor (|y| < 3 m, cols 270..329) of each band 0-5 / 5-10 / 10-20 / 20-40 / 40-100 m.  Stamps are added until the POOLED corridor precision
TP/(TP+FP) of that band equals p (`nearfield_prec.py`'s definition: exact precision on GT-seen cells), so p is directly the MAP-2 quantity.  pdrv = drivable-or-unjudged minus stamps
(so the edge feature reads true kerbs / lane edges plus the stamp borders); code = GT with stamps = sidewalk.  Modes: `--mode dose --p P`, `--mode p1` (no stamps: the GT map),
`--mode all` (psw = 255 and pdrv = 255 everywhere: a constant map).  The banked scorer reads rasters with np.load(<stub dir>/eNNN_wNNNN.npz); this driver intercepts exactly those loads.

    python -X utf8 e0h_dose.py --build-cache                                # once: gt planes of the 4,754 windows -> F:/r9e0/e0h/gt_cache.npy
    python -X utf8 e0h_dose.py --mode dose --p 0.7 --out F:/r9e0/e0h/runs/p0.70
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path("C:/Users/Admin/refcv8_pf/refcv9_mapedge")          # banked MAPPATH package (code + seals + result/), read-only
TRACE = "F:/refcv9_mapsel/trace_s0_mappath.npz"                  # MAPSEL's banked fan, progress-matched (PREREG_MAPPATH s1)
RAST = "F:/refcv9_mapsel/rast_s0"
WORK = Path("F:/r9e0/e0h")
CACHE = WORK / "gt_cache.npy"
STUB = WORK / "rast_stub"
CORR = (270, 330)                                                # corridor |y| < 3 m (nearfield_prec.py c0, c1)
BANDS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 100))        # metres along x; rows = 10 * metres
DRV = (1, 2, 3, 4, 6)
SW, EDGE, UNSEEN, NOCLS = 7, 5, 255, 0


# ------------------------------------------------------------------------------------------------- pure functions (unit-tested)
def corridor_counts(gt):
    """per band: (seen-and-sidewalk cells TP-candidates, corridor seen cells) of ONE gt plane."""
    out = []
    for a, b in BANDS:
        g = gt[a * 10:b * 10, CORR[0]:CORR[1]]
        out.append((int(((g == SW)).sum()), int((g != UNSEEN).sum())))
    return out


def draw_stamps(gts, p, stamp_cells, seed):
    """gts: iterable-indexable of gt planes [n][H, W].  Returns (stamps {k: [(r0, c0), ...]}, realized {band: (TP, FP, precision)}).
    Stamps are drawn i.i.d. (window uniform, centre uniform inside band x corridor) until the pooled FP cell count of each band reaches round(TP (1-p)/p);
    a stamp contributes only its cells that are SEEN, not GT-sidewalk and not already a false flag.  p = 1 -> no stamps."""
    n = len(gts)
    rng = np.random.default_rng(seed)
    stamps = {}
    realized = {}
    fpm = {}                                                      # k -> bool mask of false-flag cells (corridor window only, sparse-per-window)
    for bi, (a, b) in enumerate(BANDS):
        tp = sum(int((gts[k][a * 10:b * 10, CORR[0]:CORR[1]] == SW).sum()) for k in range(n))
        target = int(round(tp * (1.0 - p) / p)) if p < 1.0 else 0
        fp, guard = 0, 0
        r_lo, r_hi = a * 10, b * 10 - stamp_cells
        c_lo, c_hi = CORR[0], CORR[1] - stamp_cells
        while fp < target:
            k = int(rng.integers(0, n))
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
            g = gts[k][r0:r0 + stamp_cells, c0:c0 + stamp_cells]
            m = fpm.setdefault(k, np.zeros((1000, CORR[1] - CORR[0]), bool))
            sub = m[r0:r0 + stamp_cells, c0 - CORR[0]:c0 - CORR[0] + stamp_cells]
            new = (g != UNSEEN) & (g != SW) & ~sub
            if new.any():
                sub |= new
                fp += int(new.sum())
                stamps.setdefault(k, []).append((r0, c0))
            guard += 1
            if guard > 50_000_000:
                raise RuntimeError("stamp loop did not converge")
        realized[f"{a}-{b}"] = {"TP": tp, "FP": fp, "precision": (tp / (tp + fp)) if tp + fp else None, "target_precision": p}
    return stamps, realized


def synth_planes(gt, stamps, stamp_cells, mode):
    """one window: gt plane + its stamp list -> dict(psw, pdrv, pedge, code, gt) uint8 planes of the synthetic planner map."""
    if mode == "all":
        z = np.full(gt.shape, 255, np.uint8)
        return {"psw": z, "pdrv": z.copy(), "pedge": np.zeros_like(z), "code": np.where(gt == UNSEEN, 0, gt).astype(np.uint8), "gt": gt}
    sw = gt == SW
    fp = np.zeros(gt.shape, bool)
    for r0, c0 in stamps or ():
        blk = gt[r0:r0 + stamp_cells, c0:c0 + stamp_cells]
        fp[r0:r0 + stamp_cells, c0:c0 + stamp_cells] |= (blk != UNSEEN) & (blk != SW)
    judged_nondrv = (gt != UNSEEN) & (gt != NOCLS) & ~np.isin(gt, DRV)          # codes 5 (edge) and 7 (sidewalk)
    code = np.where(gt == UNSEEN, 0, gt).astype(np.uint8)
    code[fp] = SW
    return {"psw": (255 * (sw | fp)).astype(np.uint8), "pdrv": (255 * (~judged_nondrv & ~fp)).astype(np.uint8),
            "pedge": (255 * (gt == EDGE)).astype(np.uint8), "code": code, "gt": gt}


# ------------------------------------------------------------------------------------------------- driver
class Synth:
    """duck-types the NpzFile the banked scorer reads (`.files`, `[key]`); thread-safe (no shared mutable state)."""
    def __init__(self, gt, stamps, cells, mode, has_gt=True):
        self._gt, self._st, self._c, self._m = gt, stamps, cells, mode
        self._cache = None
        self.files = ["pdrv", "pedge", "psw", "code"] + (["gt"] if has_gt else [])      # a window without GT has no `gt` key, as in the banked rasters

    def __contains__(self, k):
        return k in self.files

    def __getitem__(self, k):
        if self._cache is None:
            self._cache = synth_planes(self._gt, self._st, self._c, self._m)
        return self._cache[k]

    def __iter__(self):
        return iter(self.files)

    def close(self):
        pass


def build_cache():
    WORK.mkdir(parents=True, exist_ok=True)
    z = np.load(TRACE, allow_pickle=False)
    e_i, ws = z["e_i"], z["ws"]
    mm = np.lib.format.open_memmap(str(CACHE), mode="w+", dtype=np.uint8, shape=(len(ws), 1000, 600))
    has = np.zeros(len(ws), np.int64)
    for k, (e, w) in enumerate(zip(e_i, ws)):
        zz = np.load(Path(RAST) / f"e{int(e):03d}_w{int(w):04d}.npz")
        if "gt" in zz.files:
            mm[k] = zz["gt"]
            has[k] = 1
        else:
            mm[k] = UNSEEN                      # no GT for this window: nothing is judged, no stamp can land
    mm.flush()
    np.save(WORK / "gt_cache_index.npy", np.stack([e_i, ws, has], 1))
    print("windows with gt", int(has.sum()), "of", len(ws))
    STUB.mkdir(parents=True, exist_ok=True)
    for e, w in zip(e_i, ws):
        (STUB / f"e{int(e):03d}_w{int(w):04d}.npz").write_bytes(b"")
    print("cache built", mm.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--mode", choices=("dose", "p1", "all"), default="dose")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--stamp-m", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.build_cache:
        return build_cache()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G = np.load(CACHE, mmap_mode="r")
    idx = np.load(WORK / "gt_cache_index.npy")
    pos = {(int(e), int(w)): k for k, (e, w, _h) in enumerate(idx)}
    hasgt = {int(k): bool(h) for k, (_e, _w, h) in enumerate(idx)}
    cells = int(round(a.stamp_m * 10))
    stamps, realized = {}, {}
    if a.mode == "dose":
        # ints seed per (p, stamp size) so every run is reproducible and independent
        stamps, realized = draw_stamps(G, a.p, cells, a.seed + int(round(a.p * 1000)) + 7 * cells)
    json.dump({"mode": a.mode, "p": a.p if a.mode == "dose" else (1.0 if a.mode == "p1" else None), "stamp_m": a.stamp_m, "seed": a.seed, "realized_corridor_precision": realized,
               "n_windows_with_stamps": len(stamps), "n_stamps": int(sum(len(v) for v in stamps.values()))}, open(out / "SYNTH_RECORD.json", "w"), indent=1)
    orig_load = np.load
    stub = str(STUB).replace("\\", "/")

    def load(path, *args, **kw):
        ps = str(path).replace("\\", "/")
        if ps.startswith(stub):
            b = os.path.basename(ps)
            e, w = int(b[1:4]), int(b[6:10])
            k = pos[(e, w)]
            return Synth(np.asarray(G[k]), stamps.get(k), cells, "all" if a.mode == "all" else "dose", hasgt[k])
        return orig_load(path, *args, **kw)
    np.load = load
    sys.path.insert(0, str(PKG))
    import mapsoft_path as MP  # noqa: E402  (banked code, UNCHANGED; its seals are checked by its own main())
    sys.argv = ["mapsoft_path.py", "--out", str(out), "--trace", TRACE, "--rast", stub]
    return MP.main()


if __name__ == "__main__":
    sys.exit(main())
