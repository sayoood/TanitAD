"""PRIVGEOM-CLEAN design calculation (MDE / power at 76 logs).  Uses ONLY (i) PRIVGEOM-SAFE's banked SEEN-log arrays (41 logs, labels already opened by that arm) and (ii) the LABEL-FREE sizes of the 76 unseen logs
(sets per log, F-stratum proposals per log, from the census `code`).  Opens NO label of any of the 76 logs, computes no outcome on them.

The estimator's variance is a function of the per-log decomposition of the paired difference:  Delta = sum_l e_l / sum_l d_l with e_l = n_a,l - n_b,l (concordant pairs of the two arms) and d_l the F-pair count of log l.
Cluster-robust (delta-method) variance  SE^2 = sum_l r_l^2 / D^2  with r_l = e_l - Delta d_l  (checked below against the registered bootstrap on the SEEN data).
Model for the 76:  Var(r_l) = a d_l + b d_l^2  (a: set-level noise, b: log-level random effect; NNLS on the 41 SEEN logs), d_l = c * fprop_l^alpha  (log-log OLS on the SEEN logs), fprop_l = F-stratum proposals of log l (label-free).
Three scalings are reported; the CONSERVATIVE one (largest SE) sizes the bars.  EVIDENCE CLASS: MEASURED inputs (SEEN), ESTIMATED SE at 76 logs.

    python pgc_design.py --out <design.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "ref")))
import numpy as np                                                                       # noqa: E402
from scipy.optimize import nnls                                                          # noqa: E402
from scipy.stats import norm                                                             # noqa: E402

import pg_lib as P                                                                       # noqa: E402

SAFE = "C:/Users/Admin/par18/privgeom"
META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
Z = 1.959964


def confirm_threshold(se: float) -> float:
    """CONFIRM iff lower CI > 0 (point > 1.96 se) AND point >= +0.02: the point must clear max(0.02, 1.96 se)."""
    return max(0.02, Z * se)


def p_confirm(delta: float, se: float) -> float:
    return float(norm.cdf((delta - confirm_threshold(se)) / se))


def p_null(delta: float, se: float) -> float:
    """NULL iff upper CI < +0.02 (point < 0.02 - 1.96 se)."""
    return float(norm.cdf((0.02 - Z * se - delta) / se))


def mde(se: float, power: float = 0.80) -> float:
    return float(confirm_threshold(se) + norm.ppf(power) * se)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cov = json.load(open(a.coverage, encoding="utf-8"))
    # ---- SEEN (banked) per-log decomposition ----------------------------------------------------------------------------------------------------
    A = np.load(f"{SAFE}/run/join/arrays.npz", allow_pickle=False)
    lab = np.load(f"{SAFE}/run/join/labels_sub.npy")
    li, code = A["log_idx"], A["code"]
    nl = int(li.max()) + 1
    an = P.Analyzer(lab, code, li, nl)
    sL = an.per_log_stats(np.load(f"{SAFE}/run/fits/L_dac_s0.npy"), "dac")
    sD = an.per_log_stats(np.load(f"{SAFE}/run/descr/fits/drop0_dac_s0.npy"), "dac")
    nL, d = sL["Q4"]
    nD, dd = sD["Q4"]
    assert np.array_equal(d, dd)
    e = nD - nL
    D = float(d.sum())
    delta = float(e.sum() / D)
    r = e - delta * d
    se_lin = float(math.sqrt((r ** 2).sum() * nl / (nl - 1)) / D)
    ci = an.diff(sD, sL, "Q4")
    se_boot = float((ci["hi"] - ci["lo"]) / (2 * Z))
    fprop = np.bincount(li, weights=(code == 4).sum(1), minlength=nl)
    m = d > 0
    # d ~ c fprop^alpha (log-log OLS)
    alpha, logc = np.polyfit(np.log(fprop[m]), np.log(d[m]), 1)
    # Var(r) = a d + b d^2 (weighted NNLS)
    x, y = d[m], r[m] ** 2
    coef, _ = nnls(np.stack([x, x ** 2], 1) / x[:, None], y / x)
    a_, b_ = float(coef[0]), float(coef[1])
    # ---- the 76 unseen logs: LABEL-FREE sizes -----------------------------------------------------------------------------------------------------
    z = np.load(META, allow_pickle=False)
    kl, codeall = z["key_log"], z["code"]
    fp76 = np.array([float((codeall[kl == n] == 4).sum()) for n in cov["logs_in_order"]])
    sets76 = np.array([cov["per_log"][n]["n_sets"] for n in cov["logs_in_order"]], float)
    fp41 = float(fprop.sum())
    d76 = np.exp(logc) * np.maximum(fp76, 1.0) ** alpha
    # rescale so that the total pairs per F proposal matches the SEEN total (the label base rate is a property of the data, not of the size)
    d76 = d76 * (D / fp41 * fp76.sum()) / d76.sum()
    D76 = float(d76.sum())
    se_model = float(math.sqrt(a_ * D76 + b_ * float((d76 ** 2).sum())) / D76)
    se_pairs = se_lin * math.sqrt(D / D76)
    se_logs = se_lin * math.sqrt(nl / len(cov["logs_in_order"]))
    se_cons = max(se_model, se_pairs, se_logs)
    # nested bootstrap of the model SE over the SEEN logs (uncertainty of a, b, alpha)
    rng = np.random.default_rng(20261019)
    sem_b = []
    for _ in range(2000):
        ii = rng.integers(0, nl, nl)
        mm = d[ii] > 0
        if mm.sum() < 8:
            continue
        try:
            al_, lc_ = np.polyfit(np.log(np.maximum(fprop[ii][mm], 1)), np.log(d[ii][mm]), 1)
            xx, yy = d[ii][mm], r[ii][mm] ** 2
            cf, _ = nnls(np.stack([xx, xx ** 2], 1) / xx[:, None], yy / xx)
            dd76 = np.exp(lc_) * np.maximum(fp76, 1.0) ** al_
            dd76 = dd76 * (D / fp41 * fp76.sum()) / dd76.sum()
            sem_b.append(math.sqrt(cf[0] * dd76.sum() + cf[1] * float((dd76 ** 2).sum())) / dd76.sum())
        except Exception:                                                                # noqa: BLE001
            continue
    sem_b = np.array(sem_b)
    out = {"schema": "privgeom-clean-design-v1", "evidence": "MEASURED inputs (PRIVGEOM-SAFE banked SEEN arrays, 41 logs, seed 0), ESTIMATED SE at 76 logs; no label of the 76 is read",
           "seen": {"n_logs": nl, "D_pairs_F": D, "delta_F_drop0_vs_L": delta, "ci_boot": [ci["lo"], ci["hi"]], "se_boot": se_boot, "se_cluster_linearised": se_lin,
                    "se_agree_ratio": se_lin / se_boot, "fit_d_vs_fprop": {"alpha": float(alpha), "c": float(math.exp(logc))}, "var_r_a": a_, "var_r_b": b_,
                    "largest_log_share_of_pairs": float(d.max() / D)},
           "unseen": {"n_logs": len(cov["logs_in_order"]), "n_sets": cov["n_sets"], "F_proposals": float(fp76.sum()), "largest_log_sets": float(sets76.max()), "largest_log_share_of_F_proposals": float(fp76.max() / fp76.sum()),
                      "predicted_D_pairs_F": D76, "predicted_largest_log_share_of_pairs": float(d76.max() / D76)},
           "se_at_76": {"model_a_b": se_model, "model_nested_boot_p05_p50_p95": [float(np.percentile(sem_b, q)) for q in (5, 50, 95)], "scale_by_pairs": se_pairs, "scale_by_logs": se_logs, "conservative_max": se_cons},
           "z": Z}
    ses = {"pairs_scaling": se_pairs, "logs_scaling": se_logs, "model_conservative": se_model, "model_nested_boot_p95": float(np.percentile(sem_b, 95))}
    grid = [0.0, 0.005, 0.010, 0.015, 0.020, 0.0276, 0.0352, 0.040]
    out["rule"] = {"confirm": "on BOTH seeds: lower CI > 0 AND point >= +0.02", "null": "on BOTH seeds: upper CI < +0.02", "normal_approximation": "CI = point +- 1.96 se, se treated as known; the two seeds are one draw of logs (Delta agrees to 0.0007 across seeds in PRIVGEOM-SAFE)"}
    out["se_variants"] = ses
    out["confirm_point_threshold_by_se"] = {k: confirm_threshold(v) for k, v in ses.items()}
    out["MDE80_confirm_by_se"] = {k: mde(v, 0.80) for k, v in ses.items()}
    out["MDE90_confirm_by_se"] = {k: mde(v, 0.90) for k, v in ses.items()}
    out["power_table"] = [{"delta": g, **{f"P_confirm[{k}]": p_confirm(g, v) for k, v in ses.items()}, **{f"P_null[{k}]": p_null(g, v) for k, v in ses.items()}} for g in grid]
    out["P_false_confirm_at_delta0_conservative_se"] = p_confirm(0.0, se_model)
    out["se_at_76"]["chosen_for_the_MDE_statement"] = se_model
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
