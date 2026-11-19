#!/usr/bin/env python
"""cv_room.py -- T0 attribution from labels only (no model, no GPU). Written by the lead, 2026-11-19. Names nothing and moves no bar.

Question: how far is the planner's privileged source (cand-label of card PREREG_V8F_PROOF_ADD40 = label gap and label closing speed AT THE TICK, lead speed kept
constant) from the true gap at 1.0 s and 2.0 s under the logged ego motion? That error is all the room a perfect forecast of the lead's motion could add to the
label-fed planner; it is set beside the read error of the lead head on the same horizon (reader of card ADD36, banked).

Forecasts of gap(j + h) from rows at or before j (join_eval141.npz; ego speed = column 0 of the bank res/cache/frames.npz, row (clip, frame)):
  cv_lead   g + v_L * h * DT - ego progress j..j+h      v_L = max(0, v_ego(j) + c)   (the formula of imag_plan.py for cand-label; ego progress by the trapezoid rule)
  cv_rel    g + c * h * DT                              (gap rate kept constant; the ego's own speed change is not known to it)
  persist   g
g, c = LabelSource.read of distrule/v8f_t1_eval_rule.py re-implemented here: closing = LS slope of the trailing run (<= 5 frames, steps <= 3.0 m, >= 3 frames) else 0.0.
Windows: label valid (0 <= gap <= 60 m) at j and at j + h, ego speed finite at j..j+h. Subset "moving": gap 5 to 50 m and ego speed >= 5 m/s at j (the H2 rule).
Interval: episode-cluster bootstrap of the mean absolute error (B 2,000, rng 0).
"""
import argparse
import io
import json

import numpy as np

DT, GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN = 0.2, 60.0, 3.0, 5, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    J = np.load(a.join, allow_pickle=True)
    Z = np.load(a.bank, allow_pickle=True)
    off, n, valid, gap = (np.asarray(J[k]) for k in ("off", "n", "lead_valid", "lead_gap"))
    gap = gap.astype(np.float64)
    clips = [str(c) for c in J["clip"]]
    spd = {(str(e), int(g)): float(v) for e, g, v in zip(Z["beid"], Z["bg"], Z["E"][:, 0])}

    def ok(r):
        g = gap[r]
        return bool(valid[r] == 1 and np.isfinite(g) and 0.0 <= g <= GAP_MAX_M)

    def read(ci, j):
        o = int(off[ci])
        r = o + j
        idx = [r]
        f = r - 1
        while f >= o and len(idx) < RUN_MAX and ok(f) and abs(gap[f + 1] - gap[f]) <= STEP_MAX_M:
            idx.append(f)
            f -= 1
        g = float(gap[r])
        if len(idx) < RUN_MIN:
            return g, 0.0, len(idx)
        idx = idx[::-1]
        G = gap[idx]
        t = np.arange(len(idx), dtype=np.float64) * DT
        tb = t.mean()
        return g, float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum()), len(idx)

    res = {"tier": "T0 (label-only attribution on logged frames; no model is read; never driving performance)",
           "strategic": "STRATEGIC: NOT COMPUTED (n = 0; label-only attribution of the lead gap, no strategic instrument)",
           "names": "none (descriptive attribution, decides no name and moves no bar)", "horizons": {}}
    rng = np.random.default_rng(0)
    for h in (5, 10):
        rows = []
        for ci, c in enumerate(clips):
            for j in range(int(n[ci]) - h):
                r = int(off[ci]) + j
                if not (ok(r) and ok(r + h)):
                    continue
                v = np.array([spd.get((c, j + k), np.nan) for k in range(h + 1)], np.float64)
                if not np.isfinite(v).all():
                    continue
                g, cl, run = read(ci, j)
                prog = float((0.5 * (v[:-1] + v[1:]) * DT).sum())
                vL = max(0.0, v[0] + cl)
                y = float(gap[r + h])
                rows.append((ci, g, cl, run, v[0], y, g + vL * h * DT - prog, g + cl * h * DT, g, v[-1] - v[0]))
        R = np.array(rows, np.float64)
        ep = R[:, 0].astype(np.int64)
        y = R[:, 5]
        mov = (R[:, 1] >= 5.0) & (R[:, 1] <= 50.0) & (R[:, 4] >= 5.0)
        out = {"n_windows": int(len(R)), "n_episodes": int(len(np.unique(ep))), "n_windows_moving": int(mov.sum()), "n_episodes_moving": int(len(np.unique(ep[mov]))),
               "share_closing_from_a_run_of_3_or_more": float((R[:, 3] >= RUN_MIN).mean())}
        for sub, m in (("all", np.ones(len(R), bool)), ("moving", mov)):
            S = {}
            for nm, col in (("cv_lead", 6), ("cv_rel", 7), ("persist", 8)):
                e = np.abs(R[m, col] - y[m])
                eps = np.unique(ep[m])
                per = np.array([[e[ep[m] == k].sum(), (ep[m] == k).sum()] for k in eps])
                bs = []
                for _ in range(a.B):
                    d = per[rng.integers(0, len(eps), len(eps))]
                    bs.append(d[:, 0].sum() / d[:, 1].sum())
                S[nm] = {"mae_m": round(float(e.mean()), 4), "interval": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                         "median_abs_m": round(float(np.median(e)), 4), "p90_abs_m": round(float(np.percentile(e, 90)), 4),
                         "share_above_2m": round(float((e > 2.0).mean()), 4), "bias_m": round(float((R[m, col] - y[m]).mean()), 4)}
            S["true_gap_change_abs_mean_m"] = round(float(np.abs(y[m] - R[m, 8]).mean()), 4)
            S["ego_speed_change_abs_mean_mps"] = round(float(np.abs(R[m, 9]).mean()), 4)
            out[sub] = S
        res["horizons"]["%.1f_s" % (h * DT)] = out
        print("== horizon %.1f s: %d windows of %d episodes (moving: %d of %d)" % (h * DT, out["n_windows"], out["n_episodes"], out["n_windows_moving"], out["n_episodes_moving"]))
        for sub in ("all", "moving"):
            for nm in ("cv_lead", "cv_rel", "persist"):
                print("  ", sub, nm, json.dumps(out[sub][nm]))
            print("  ", sub, "true gap change", out[sub]["true_gap_change_abs_mean_m"], "m; ego speed change", out[sub]["ego_speed_change_abs_mean_mps"], "m/s")
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("ZZ CV-ROOM DONE")


if __name__ == "__main__":
    main()
