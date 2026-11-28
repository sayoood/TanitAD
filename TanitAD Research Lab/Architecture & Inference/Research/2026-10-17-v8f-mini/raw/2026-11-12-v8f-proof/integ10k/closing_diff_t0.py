# closing_diff_t0.py : T0 feasibility read after card PREREG_V8F_PROOF_ADD49 ended NOT NAMED on control h2 (the reader's trained closing output: Pearson r 0.27 / 0.28
# against the label closing speed, mean absolute error not below that of the constant 0). Question: is the CHANGE of the reader's own gap read over a baseline of k frames
# a usable closing speed? Rows: the banked reader rows of card ADD46 (rows_LR16_s<k>.npz: per frame presence probability p, gap read g, label presence, label gap; clips
# held out from the reader's training; the flag `fit` marks the clips the maps were fitted on). These are training-corpus clips, not the evaluation episodes, so choosing
# the baseline here is a calibration and no selection on the evaluation windows. Tier T0 (decodability on logged frames); never driving performance. Names nothing.
# Convention written into TIMES_NOTE.md before this ran: per reader seed and baseline k in (5, 10, 15, 25) frames (0.2 s per frame), on rows where the label has a lead
# within 60 m at t and t-k and the reader is present (p >= 0.5) at both: L = (label gap(t) - label gap(t-k)) / (0.2 k), R = (g(t) - g(t-k)) / (0.2 k).
# The baseline is chosen by the highest Pearson r on the FIT clips; on the CHECK clips it is read: r, MAE of R, MAE of the shrunk read s*R (s = least-squares slope on
# the fit clips), MAE of the constant 0. FEASIBLE if at both reader seeds r on the check clips is at least 0.5 and the shrunk MAE is at least 10 percent below the MAE of 0.
# Usage (pod): python closing_diff_t0.py --rows-dir /workspace/v8proof/integ10k/rows --out CLOSING_DIFF_T0.json
import argparse
import json

import numpy as np

DT, G_MAX, P_MIN, KS = 0.2, 60.0, 0.5, (5, 10, 15, 25)
R_BAR, GAIN_BAR = 0.5, 0.10


def pairs(z, k):
    ci, t = np.asarray(z["ci"], np.int64), np.asarray(z["t"], np.int64)
    key = ci * 100000 + t
    pos = {int(v): i for i, v in enumerate(key)}
    i1 = np.arange(len(key))
    i0 = np.array([pos.get(int(v) - k, -1) for v in key])
    ok = i0 >= 0
    i1, i0 = i1[ok], i0[ok]
    lp, lg, p, g = np.asarray(z["lab_pres"]).astype(bool), np.asarray(z["lab_gap"], np.float64), np.asarray(z["p"], np.float64), np.asarray(z["g"], np.float64)
    m = lp[i1] & lp[i0] & np.isfinite(lg[i1]) & np.isfinite(lg[i0]) & (lg[i1] <= G_MAX) & (lg[i0] <= G_MAX) & (p[i1] >= P_MIN) & (p[i0] >= P_MIN)
    i1, i0 = i1[m], i0[m]
    L = (lg[i1] - lg[i0]) / (DT * k)
    R = (g[i1] - g[i0]) / (DT * k)
    return L, R, np.asarray(z["fit"]).astype(bool)[i1], ci[i1]


def stats(L, R, s=None):
    if len(L) < 20:
        return {"n": int(len(L))}
    out = {"n": int(len(L)), "pearson_r": float(np.corrcoef(L, R)[0, 1]), "mae_raw_mps": float(np.abs(R - L).mean()), "mae_zero_mps": float(np.abs(L).mean()),
           "std_label_mps": float(L.std()), "std_read_mps": float(R.std())}
    if s is not None:
        out["slope_used"] = float(s)
        out["mae_shrunk_mps"] = float(np.abs(s * R - L).mean())
        out["gain_of_shrunk_over_zero"] = float(1.0 - out["mae_shrunk_mps"] / out["mae_zero_mps"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"what": "T0 feasibility of a closing speed from the change of the reader's gap read; names nothing", "tier": "T0", "bars": {"r_check": R_BAR, "gain_of_shrunk_over_zero": GAIN_BAR}, "per_reader_seed": {}}
    feas = []
    for seed in (0, 1):
        z = np.load("%s/rows_LR16_s%d.npz" % (a.rows_dir, seed), allow_pickle=False)
        S = {"per_k": {}}
        best, best_r = None, -9.0
        for k in KS:
            L, R, fit, ci = pairs(z, k)
            f = stats(L[fit], R[fit])
            s = float((L[fit] * R[fit]).sum() / max(1e-9, (R[fit] * R[fit]).sum())) if fit.sum() >= 20 else None
            c = stats(L[~fit], R[~fit], s)
            S["per_k"][str(k)] = {"fit_clips": f, "check_clips": c, "n_check_clips": int(len(np.unique(ci[~fit])))}
            if f.get("pearson_r", -9.0) > best_r:
                best, best_r = k, f["pearson_r"]
            print("s%d k=%2d (%.1f s) fit: n %d r %.3f | check: n %d r %s MAE raw %s shrunk %s zero %s gain %s (clips %d)" % (
                seed, k, DT * k, f["n"], f.get("pearson_r", float("nan")), c["n"], c.get("pearson_r"), c.get("mae_raw_mps"), c.get("mae_shrunk_mps"), c.get("mae_zero_mps"),
                c.get("gain_of_shrunk_over_zero"), S["per_k"][str(k)]["n_check_clips"]))
        c = S["per_k"][str(best)]["check_clips"]
        S["chosen_k"] = best
        S["feasible"] = bool(c.get("pearson_r", 0.0) >= R_BAR and c.get("gain_of_shrunk_over_zero", 0.0) >= GAIN_BAR)
        feas.append(S["feasible"])
        res["per_reader_seed"][str(seed)] = S
        print("s%d chosen k %d: check r %s gain %s -> feasible %s" % (seed, best, c.get("pearson_r"), c.get("gain_of_shrunk_over_zero"), S["feasible"]))
    res["FEASIBLE"] = bool(all(feas))
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ CLOSING-DIFF-T0 %s" % ("FEASIBLE" if res["FEASIBLE"] else "NOT-FEASIBLE"))


if __name__ == "__main__":
    main()
