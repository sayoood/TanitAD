"""perc8_rows.py -- per-window GREEDY ROWS of a banked perception pass, built ONCE with the programme's own matcher
(`tanitad.eval.detection_metrics.greedy_rows`) so every later number is derived from the same rows the trainer's own
`summarise` would pool.  CPU only.  Multiprocessing; each worker re-reads the pickle itself (works under fork and spawn).

    python perc8_rows.py --pkl <perc_det.pkl> --out <rows.npz> [--workers 8] [--nms-radius 2.5 --nms-pfloor 0.05]

The pack of each window is `rec['box3d'][0]` (the same slot `perc_score.py` reads).  With `--nms-radius` the per-window pack
is first passed through `detection_nms.nms_pack` (F4b, read-only; a NEW pack) and the rows are those of the NMS'd pack.

Output npz (all rows with kind >= 0, i.e. DontCare removed exactly as `detection_metrics._ap` does):
  win_sha, win_ws                      [W]   window identity (sha12 str, ws int), in pickle order
  npos[w, c, b]                        [W, 11, 4]  positives by (class: 0=all, 1..10 = class index + 1; band: 0=all, 1..3 = bands)
  for every threshold index t (0..3 = 0.5, 1, 2, 4 m) and "scope" s in {all, cls}:
     t{t}_all_{conf,hit,band,win}    rows of the all-class greedy pass
     t{t}_cls_{conf,hit,band,win,c}  rows of the per-class passes (c = class index), concatenated over classes
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np

THR = (0.5, 1.0, 2.0, 4.0)
NCLS = 10
BAND_CODE = {None: -1, "0_20": 0, "20_40": 1, "40_60": 2}

_RECS = None
_NMS = None


def _init(pkl, nms_radius, nms_pfloor):
    global _RECS, _NMS
    det = pickle.load(open(pkl, "rb"))
    _RECS = [(r["sha12"], int(r["ws"]), r["box3d"][0]) for r in det if r.get("box3d")]
    _NMS = None if nms_radius is None else (float(nms_radius), float(nms_pfloor))


def npos_of(pk):
    """[10, 4] positives per (class-or-all, band-or-all) -- `detection_metrics.summarise`'s own counting rule."""
    from tanitad.eval import detection_metrics as dm
    out = np.zeros((1 + NCLS, 4), np.int64)
    for gxy, gc, p in zip(pk["gt_xy"], pk["gt_cls"], pk["pos"]):
        if not p:
            continue
        bd = dm._band_of(float(gxy[0]))
        bi = 0 if bd is None else 1 + [b[0] for b in dm.BANDS_X_M].index(bd)
        for ci in (0, 1 + int(gc)):
            out[ci, 0] += 1
            if bd is not None:
                out[ci, bi] += 1
    return out


def _work(span):
    from tanitad.eval import detection_metrics as dm
    from tanitad.eval import detection_nms as nms
    lo, hi = span
    res = []
    for i in range(lo, hi):
        sha, ws, pk = _RECS[i]
        if _NMS is not None:
            pk = nms.nms_pack(pk, _NMS[0], _NMS[1])
        npo = npos_of(_RECS[i][2])          # positives never change under NMS (it only drops slots)
        per = {}
        for ti, t in enumerate(THR):
            for si, cls in enumerate([None] + list(range(NCLS))):
                rows = [r for r in dm.greedy_rows(pk, t, cls) if r[1] >= 0]
                if not rows:
                    continue
                per[(ti, si)] = (np.array([r[0] for r in rows], np.float64),
                                 np.array([r[1] for r in rows], np.int8),
                                 np.array([BAND_CODE[r[2]] for r in rows], np.int8))
        res.append((i, npo, per))
    return res


def build(pkl, workers, nms_radius=None, nms_pfloor=0.05, chunk=40):
    _init(pkl, nms_radius, nms_pfloor)
    n = len(_RECS)
    spans = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    t0 = time.time()
    if workers <= 1:
        parts = [_work(s) for s in spans]
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init, initargs=(pkl, nms_radius, nms_pfloor)) as pool:
            parts = []
            for k, r in enumerate(pool.imap(_work, spans)):
                parts.append(r)
                if k % 10 == 0:
                    print(f"[rows] {k + 1}/{len(spans)} chunks {time.time() - t0:.0f}s", flush=True)
    flat = sorted((x for p in parts for x in p), key=lambda x: x[0])
    assert [x[0] for x in flat] == list(range(n))
    out = {"win_sha": np.array([r[0] for r in _RECS]), "win_ws": np.array([r[1] for r in _RECS], np.int64),
           "npos": np.stack([x[1] for x in flat])}
    for ti in range(len(THR)):
        for si, name in ((0, "all"), (1, "cls")):
            conf, hit, band, win, cl = [], [], [], [], []
            for (i, _, per) in flat:
                for sj in ([0] if si == 0 else range(1, 1 + NCLS)):
                    if (ti, sj) in per:
                        c_, h_, b_ = per[(ti, sj)]
                        conf.append(c_)
                        hit.append(h_)
                        band.append(b_)
                        win.append(np.full(len(c_), i, np.int32))
                        if si == 1:
                            cl.append(np.full(len(c_), sj - 1, np.int8))
            out[f"t{ti}_{name}_conf"] = np.concatenate(conf) if conf else np.zeros(0)
            out[f"t{ti}_{name}_hit"] = np.concatenate(hit) if hit else np.zeros(0, np.int8)
            out[f"t{ti}_{name}_band"] = np.concatenate(band) if band else np.zeros(0, np.int8)
            out[f"t{ti}_{name}_win"] = np.concatenate(win) if win else np.zeros(0, np.int32)
            if si == 1:
                out[f"t{ti}_{name}_c"] = np.concatenate(cl) if cl else np.zeros(0, np.int8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nms-radius", type=float, default=None)
    ap.add_argument("--nms-pfloor", type=float, default=0.05)
    a = ap.parse_args()
    t0 = time.time()
    out = build(a.pkl, a.workers, a.nms_radius, a.nms_pfloor)
    np.savez_compressed(a.out, **out)
    print(f"[rows] done {a.out}: {len(out['win_sha'])} windows, "
          f"{sum(len(v) for k, v in out.items() if k.endswith('_conf'))} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
