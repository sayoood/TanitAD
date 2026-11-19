#!/usr/bin/env python
"""read_parts.py -- T0, descriptive (names nothing): the two parts of the lead head's present read on the reader's windows, each against its label.
  gap part      head gap at t        against the label gap at t
  closing part  head closing speed   against the label closing speed at t (LS slope of the trailing label run, the rule of LabelSource.read)
and what each part is worth for the 2.0 s forecast: the four mixes (label or head gap) x (label or head closing), gap + closing * 2.0 against the label gap at t + 2.0 s.
Windows: lead at t and at t + 2.0 s, a trailing label run of at least 3 frames (so that the label closing speed is a slope, not the 0.0 fallback).
Interval: episode-cluster bootstrap (B 2,000, rng 0). Written by the lead, 2026-11-19.
"""
import argparse
import io
import json

import numpy as np

DT, GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN, H = 0.2, 60.0, 3.0, 5, 3, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    J = np.load(a.join, allow_pickle=True)
    off, n, valid = (np.asarray(J[k]) for k in ("off", "n", "lead_valid"))
    gap = np.asarray(J["lead_gap"], np.float64)
    cidx = {str(c): i for i, c in enumerate(J["clip"])}

    def ok(r):
        g = gap[r]
        return bool(valid[r] == 1 and np.isfinite(g) and 0.0 <= g <= GAP_MAX_M)

    def closing(ci, j):
        o = int(off[ci])
        r = o + j
        idx = [r]
        f = r - 1
        while f >= o and len(idx) < RUN_MAX and ok(f) and abs(gap[f + 1] - gap[f]) <= STEP_MAX_M:
            idx.append(f)
            f -= 1
        if len(idx) < RUN_MIN:
            return None
        idx = idx[::-1]
        G = gap[idx]
        t = np.arange(len(idx), dtype=np.float64) * DT
        tb = t.mean()
        return float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())

    res = {"tier": "T0 (descriptive read of the lead head on logged frames; never driving performance)", "names": "none",
           "strategic": "STRATEGIC: NOT COMPUTED (n = 0; read of the lead gap and closing speed, no strategic instrument)", "dumps": {}}
    for dp in a.dumps:
        d = np.load(dp, allow_pickle=True)
        eid = d["eid"].astype(str)
        rows = []
        for i, (e, j) in enumerate(zip(eid, d["g0"].astype(np.int64))):
            if e not in cidx:
                continue
            ci = cidx[e]
            gt, g2 = float(d["gap_t"][i]), float(d["gap_t2"][i])
            hg, hc = float(d["present"][i, 1]), float(d["present"][i, 2])
            if not (np.isfinite(gt) and np.isfinite(g2) and np.isfinite(hg) and np.isfinite(hc)):
                continue
            if not (0 <= j < int(n[ci]) and abs(gap[int(off[ci]) + int(j)] - gt) < 1e-3):
                continue
            c = closing(ci, int(j))
            if c is None:
                continue
            rows.append((ci, gt, g2, c, hg, hc))
        R = np.array(rows, np.float64)
        ep = R[:, 0].astype(np.int64)
        gt, g2, c, hg, hc = R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5]
        eps = np.unique(ep)
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(eps), len(eps)) for _ in range(a.B)]
        cnt = np.array([(ep == k).sum() for k in eps], np.float64)

        def mae_ci(e):
            s = np.array([e[ep == k].sum() for k in eps])
            bs = [s[dr].sum() / cnt[dr].sum() for dr in draws]
            return {"mae": round(float(e.mean()), 4), "interval": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "median": round(float(np.median(e)), 4)}

        def r2(y, p):
            return round(float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)
        out = {"n_windows": int(len(R)), "n_episodes": int(len(eps)),
               "gap_part": dict(mae_ci(np.abs(hg - gt)), corr=round(float(np.corrcoef(hg, gt)[0, 1]), 4), r2=r2(gt, hg), bias=round(float((hg - gt).mean()), 4),
                                label_sd=round(float(gt.std()), 4)),
               "closing_part": dict(mae_ci(np.abs(hc - c)), corr=round(float(np.corrcoef(hc, c)[0, 1]), 4), r2=r2(c, hc), bias=round(float((hc - c).mean()), 4),
                                    label_sd=round(float(c.std()), 4), head_sd=round(float(hc.std()), 4), mae_of_zero=round(float(np.abs(c).mean()), 4)),
               "forecast_2s_mae_m": {"label_gap+label_closing": mae_ci(np.abs(gt + c * H * DT - g2)), "label_gap+head_closing": mae_ci(np.abs(gt + hc * H * DT - g2)),
                                     "head_gap+label_closing": mae_ci(np.abs(hg + c * H * DT - g2)), "head_gap+head_closing": mae_ci(np.abs(hg + hc * H * DT - g2)),
                                     "label_gap_held": mae_ci(np.abs(gt - g2)), "head_gap_held": mae_ci(np.abs(hg - g2))}}
        nm = dp.replace("\\", "/").split("/")[-1]
        res["dumps"][nm] = out
        print("== %s: %d windows of %d episodes" % (nm, out["n_windows"], out["n_episodes"]))
        print("   gap part    ", json.dumps(out["gap_part"]))
        print("   closing part", json.dumps(out["closing_part"]))
        for k, v in out["forecast_2s_mae_m"].items():
            print("   2.0 s %-26s %s" % (k, json.dumps(v)))
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("ZZ READ-PARTS DONE")


if __name__ == "__main__":
    main()
