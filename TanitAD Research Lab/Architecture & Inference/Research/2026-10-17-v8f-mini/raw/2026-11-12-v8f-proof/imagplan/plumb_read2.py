#!/usr/bin/env python
"""plumb_read2.py -- controls p4b, p5b and p7 of card PREREG_V8F_PROOF_ADD40 in the read forms of note N2, from the logs of one checkpoint. Written by the lead.
Reads no unsafe rate, no position error and no family value.

p4b: candidate actions of the logged future path (c = 1.0, changed tool with the yaw bound) against the corpus' logged actions on the full window set:
     yaw correlation >= 0.99 over all steps; acceleration correlation >= 0.99 over steps 2..10 (the first step beside); median |acc difference| <= 0.1 m/s2.
p5b: response reproduced. Over active ticks and candidates whose progress at 2.0 s lies 1.0 to 4.0 m behind the plan's: slope through the origin of
     (imagined gap of the candidate - imagined gap of the plan) on (progress of the plan - progress of the candidate) at 2.0 s.
     --responds yes: bar slope >= 0.5;  --responds no: bar slope < 0.25;  --responds unknown: reported, no bar.
p7:  counts; with --cv-log the active sets of cand-cv and cand-imag are the same.
Beside: ticks whose plan exceeds 60 m/s at a check horizon; ticks and steps with a bounded yaw rate.
"""
import argparse
import io
import json
import os

import numpy as np


def corr(x, y):
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def slope0(dg, da):
    return float((dg * da).sum() / (da * da).sum()) if len(da) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p45", required=True)
    ap.add_argument("--imag-log", required=True)
    ap.add_argument("--cv-log", default="")
    ap.add_argument("--responds", choices=("yes", "no", "unknown"), default="unknown")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = np.load(a.p45, allow_pickle=True)
    L = np.load(a.imag_log, allow_pickle=True)
    res = {"card": "PREREG_V8F_PROOF_ADD40 + notes N1, N2", "files": {k: os.path.basename(str(v)) for k, v in vars(a).items() if v}}

    # ---- p4b
    c, g = np.asarray(P["act_gt"], np.float64), np.asarray(P["act_cache"], np.float64)
    ok = np.isfinite(c).all((1, 2)) & np.isfinite(g).all((1, 2))
    c, g = c[ok], g[ok]
    p4 = {"n_windows": int(ok.sum()), "corr_yaw_all_steps": corr(c[..., 1], g[..., 1]), "corr_acc_steps_2_10": corr(c[:, 1:, 0], g[:, 1:, 0]),
          "corr_acc_step_1": corr(c[:, 0, 0], g[:, 0, 0]), "corr_acc_all_steps": corr(c[..., 0], g[..., 0]),
          "median_abs_diff_acc": float(np.median(np.abs(c[..., 0] - g[..., 0]))), "median_abs_diff_yaw": float(np.median(np.abs(c[..., 1] - g[..., 1]))),
          "max_abs_yaw_candidate": float(np.abs(c[..., 1]).max()), "max_abs_yaw_logged": float(np.abs(g[..., 1]).max())}
    p4["PASS"] = bool(p4["corr_yaw_all_steps"] >= 0.99 and p4["corr_acc_steps_2_10"] >= 0.99 and p4["median_abs_diff_acc"] <= 0.1)
    res["p4b"] = p4

    # ---- p5b
    cs = np.asarray(L["cand_set"], np.float64)
    i1 = int(np.flatnonzero(np.isclose(cs, 1.0))[0])
    act = np.asarray(L["active"]).astype(bool)
    ah = np.asarray(L["a_h"], np.float64)[act][:, :, 1]              # [rows, C] progress at 2.0 s
    gi = np.asarray(L["imag_raw"], np.float64)[act][:, :, 1]         # [rows, C] imagined gap at 2.0 s
    ci = np.asarray(L["ci"]).astype(int)[act]
    da = ah[:, [i1]] - ah
    dg = gi - gi[:, [i1]]
    use = (da >= 1.0) & (da <= 4.0) & np.isfinite(dg)
    use[:, i1] = False
    b = slope0(dg[use], da[use])
    rng = np.random.default_rng(0)
    clips = np.unique(ci)
    boots = []
    rows_of = {k: np.flatnonzero(ci == k) for k in clips}
    for _ in range(2000):
        pick = rng.choice(clips, len(clips), replace=True)
        idx = np.concatenate([rows_of[k] for k in pick])
        u = use[idx]
        boots.append(slope0(dg[idx][u], da[idx][u]))
    p5 = {"n_active_ticks": int(act.sum()), "n_pairs": int(use.sum()), "n_ticks_with_a_pair": int(use.any(1).sum()), "slope_T2.0": b,
          "ci_episodes": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "responds": a.responds,
          "mean_dg_m": float(dg[use].mean()) if use.any() else None, "mean_da_m": float(da[use].mean()) if use.any() else None,
          "imag_gap_plan_T2.0_median_m": float(np.nanmedian(gi[:, i1])), "n_nonfinite_ticks": int((np.asarray(L["n_nonfinite"])[act] > 0).sum())}
    p5["PASS"] = None if a.responds == "unknown" else bool(b >= 0.5 if a.responds == "yes" else b < 0.25)
    res["p5b"] = p5

    # ---- p7 and beside
    lci, lt0, lk = np.asarray(L["ci"]).astype(int), np.asarray(L["t0"]).astype(int), np.asarray(L["k"]).astype(int)
    p7 = {"rows": int(len(lci)), "unique_window_ticks": len(set(zip(lci.tolist(), lt0.tolist(), lk.tolist()))), "windows": len(set(zip(lci.tolist(), lt0.tolist()))),
          "tick_steps": sorted(set(lk.tolist())), "active_rows": int(act.sum()), "windows_p45": int(len(P["ci"]))}
    p7["PASS_counts"] = bool(p7["windows"] == 2399 and p7["rows"] == p7["unique_window_ticks"] == 3 * 2399)
    if a.cv_log:
        C = np.load(a.cv_log, allow_pickle=True)
        same = bool(len(C["ci"]) == len(lci) and (np.asarray(C["ci"]) == L["ci"]).all() and (np.asarray(C["t0"]) == L["t0"]).all() and (np.asarray(C["k"]) == L["k"]).all())
        p7["active_equal_all_ticks"] = bool(same and (np.asarray(C["active"]) == np.asarray(L["active"])).all())
    res["p7"] = p7
    vh = np.asarray(L["v_h"], np.float64)[act][:, i1, :]
    yl = np.asarray(L["n_yaw_lim"]) if "n_yaw_lim" in L.files else None
    res["beside"] = {"active_ticks_with_plan_speed_above_60": int((vh > 60.0).any(1).sum()),
                     "ticks_with_a_bounded_yaw_step": (int((yl > 0).sum()) if yl is not None else None), "bounded_yaw_steps": (int(yl.sum()) if yl is not None else None),
                     "chosen_c_histogram_NOT_READ": "left out on purpose: the choice is an outcome"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    r4 = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    print("p4b", json.dumps(r4(p4)))
    print("p5b", json.dumps({k: ([round(x, 3) for x in v] if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v)) for k, v in p5.items()}))
    print("p7", json.dumps(p7))
    print("beside", json.dumps(res["beside"]))


if __name__ == "__main__":
    main()
