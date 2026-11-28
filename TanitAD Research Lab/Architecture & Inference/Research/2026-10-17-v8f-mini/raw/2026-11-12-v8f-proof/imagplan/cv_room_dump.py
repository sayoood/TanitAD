#!/usr/bin/env python
"""cv_room_dump.py -- T0 attribution on the windows of the imagination reader (card ADD36 dumps). Written by the lead, 2026-11-19. Names nothing, moves no bar.

One window set, one label (the lead gap at t + 2.0 s), eight forecasts set side by side:
  from labels at or before t (privileged):  persist = gap_t ; cv_rel = gap_t + c * 2.0 ; cv_lead = gap_t + max(0, v_ego + c) * 2.0 - ego progress (the planner's cand-label)
  from the lead head of the arm:            present = head gap at t ; present_cv = head gap + head closing * 2.0 ; rollout = head on the rolled-out state (logged
                                            actions) ; true_future = head on the true future state
c = LS slope of the trailing label run (LabelSource.read of distrule/v8f_t1_eval_rule.py). Windows: all eight finite. The frame index of the tick in the join is
found by matching gap_t (the candidate columns of the dump are tried and the match rate is printed; under 0.99 the script stops).
Interval: episode-cluster bootstrap (B 2,000, rng 0) of the mean absolute error and of paired differences of absolute errors.
"""
import argparse
import io
import json

import numpy as np

DT, GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN, H = 0.2, 60.0, 3.0, 5, 3, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    J = np.load(a.join, allow_pickle=True)
    Z = np.load(a.bank, allow_pickle=True)
    off, n, valid = (np.asarray(J[k]) for k in ("off", "n", "lead_valid"))
    gap = np.asarray(J["lead_gap"], np.float64)
    cidx = {str(c): i for i, c in enumerate(J["clip"])}
    spd = {(str(e), int(g)): float(v) for e, g, v in zip(Z["beid"], Z["bg"], Z["E"][:, 0])}

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
            return 0.0
        idx = idx[::-1]
        G = gap[idx]
        t = np.arange(len(idx), dtype=np.float64) * DT
        tb = t.mean()
        return float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())

    res = {"tier": "T0 (attribution on logged frames; never driving performance)",
           "strategic": "STRATEGIC: NOT COMPUTED (n = 0; attribution of the lead gap forecast, no strategic instrument)",
           "names": "none (descriptive attribution, decides no name and moves no bar)", "dumps": {}}
    NAMES = ("persist", "cv_rel", "cv_lead", "present", "present_cv", "rollout", "true_future")
    PAIRS = (("true_future", "cv_lead"), ("rollout", "cv_lead"), ("present_cv", "cv_lead"), ("rollout", "present_cv"), ("rollout", "present"), ("cv_lead", "persist"))
    for dp in a.dumps:
        d = np.load(dp, allow_pickle=True)
        eid = d["eid"].astype(str)
        gt, g2 = d["gap_t"].astype(np.float64), d["gap_t2"].astype(np.float64)
        best = None
        for col in ("t", "row", "g0"):
            jj = d[col].astype(np.int64)
            hit = tot = 0
            for e, j, g in zip(eid, jj, gt):
                if not np.isfinite(g) or e not in cidx:
                    continue
                ci = cidx[e]
                tot += 1
                if 0 <= j < int(n[ci]) and np.isfinite(gap[int(off[ci]) + j]) and abs(gap[int(off[ci]) + j] - g) < 1e-3:
                    hit += 1
            rate = hit / max(tot, 1)
            if best is None or rate > best[1]:
                best = (col, rate, tot)
        print("== %s: tick frame column '%s', label match rate %.4f over %d windows" % (dp.replace("\\", "/").split("/")[-1], best[0], best[1], best[2]))
        if best[1] < 0.99:
            print("ZZ CV-ROOM-DUMP STOP: no dump column maps the tick to the join at a rate of 0.99")
            raise SystemExit(3)
        jj = d[best[0]].astype(np.int64)
        rows = []
        for i, (e, j) in enumerate(zip(eid, jj)):
            if e not in cidx or not (np.isfinite(gt[i]) and np.isfinite(g2[i])):
                continue
            ci = cidx[e]
            v = np.array([spd.get((e, int(j) + k), np.nan) for k in range(H + 1)], np.float64)
            if not np.isfinite(v).all():
                continue
            c = closing(ci, int(j))
            prog = float((0.5 * (v[:-1] + v[1:]) * DT).sum())
            f = [gt[i], gt[i] + c * H * DT, gt[i] + max(0.0, v[0] + c) * H * DT - prog, float(d["present"][i, 1]), float(d["present"][i, 1] + d["present"][i, 2] * H * DT),
                 float(d["roll_logged"][i, 1, 1]), float(d["true2"][i, 1])]
            if np.isfinite(f).all():
                rows.append([ci, g2[i]] + f)
        R = np.array(rows, np.float64)
        ep = R[:, 0].astype(np.int64)
        y = R[:, 1]
        E = {nm: np.abs(R[:, 2 + k] - y) for k, nm in enumerate(NAMES)}
        eps = np.unique(ep)
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(eps), len(eps)) for _ in range(a.B)]
        sums = {nm: np.array([E[nm][ep == k].sum() for k in eps]) for nm in NAMES}
        cnt = np.array([(ep == k).sum() for k in eps], np.float64)
        out = {"n_windows": int(len(R)), "n_episodes": int(len(eps)), "tick_column": best[0], "mae_m": {}, "paired_abs_error_difference_m": {}}
        for nm in NAMES:
            bs = [sums[nm][dr].sum() / cnt[dr].sum() for dr in draws]
            out["mae_m"][nm] = {"mae": round(float(E[nm].mean()), 4), "interval": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                "median": round(float(np.median(E[nm])), 4)}
        for x, z in PAIRS:
            bs = [(sums[x][dr].sum() - sums[z][dr].sum()) / cnt[dr].sum() for dr in draws]
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            out["paired_abs_error_difference_m"]["%s_minus_%s" % (x, z)] = {"D": round(float((E[x] - E[z]).mean()), 4), "interval": [round(lo, 4), round(hi, 4)],
                                                                             "excludes_0": bool(lo > 0 or hi < 0)}
        res["dumps"][dp.replace("\\", "/").split("/")[-1]] = out
        print("   windows %d of %d episodes" % (out["n_windows"], out["n_episodes"]))
        for nm in NAMES:
            print("   mae %-12s %s" % (nm, json.dumps(out["mae_m"][nm])))
        for k, v in out["paired_abs_error_difference_m"].items():
            print("   diff %-28s %s" % (k, json.dumps(v)))
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("ZZ CV-ROOM-DUMP DONE")


if __name__ == "__main__":
    main()
