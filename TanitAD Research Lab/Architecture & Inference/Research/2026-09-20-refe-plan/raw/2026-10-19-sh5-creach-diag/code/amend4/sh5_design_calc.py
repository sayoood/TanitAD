"""Design-stage calculations for the SH5 prereg (CPU, numpy; they read PRIVGEOM-SAFE's banked arrays READ-ONLY and produce NO SH5 outcome):

    sh5_design_calc.py plant   --out F    # calibrates the C-POWER plant: the noise sigma at which an IDEAL reader of the planted noisy signed clearances gains +0.03 F-AUC over P2's DAC row
    sh5_design_calc.py mde     --out F    # the minimum detectable effect of the registered rule on the full (4,941-set) and the local-ctx (816-token) analysis sets, from the banked sign-clean arm
    sh5_design_calc.py atest   --out F    # the sign-admissibility test (sh5_admit) of the 9 key-time |clearance| target block, with its planted controls (+ the descriptive MLP recomputer)

DISCLOSURE (prereg s0): these use the labels that PRIVGEOM-SAFE already opened (its join arrays); they fix the DESIGN (plant noise, power statement, target admissibility) before SH5 trains anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402

PG = "C:/Users/Admin/par18/privgeom"


def load_pg():
    z = np.load(f"{PG}/run/join/arrays.npz", allow_pickle=True)
    lab = np.load(f"{PG}/run/join/labels_sub.npy")
    return {k: z[k] for k in z.files}, lab


def _auc_point(FL, z, lab_dac, code, stratum="Q4"):
    st = FL.pair_stats(z, lab_dac == 0.0, lab_dac == 1.0, code, names=(stratum,))
    n, d = st[stratum]
    return float(n.sum() / d.sum())


def plant_scalar(sc_key: np.ndarray, sigma: float, token_strs: np.ndarray) -> np.ndarray:
    """the IDEAL reader's score of the weak plant: min over the 9 key times of (signed clearance + N(0, sigma)), the noise drawn per TOKEN exactly as the trainer draws it (`L.plant_noise`)."""
    noise = np.zeros(sc_key.shape, np.float64)
    for t in np.unique(token_strs):
        rows = np.flatnonzero(token_strs == t)
        noise[rows] = L.plant_noise(str(t), (len(rows),) + sc_key.shape[1:], sigma)
    return (sc_key.astype(np.float64) + noise).min(-1)


def cmd_plant(a):
    FL, PK = L.refs()
    A, lab = load_pg()
    lab_dac = lab[..., L.DAC_COL]
    zL = A["z6"][..., L.DAC_COL]
    m = np.load("C:/Users/Admin/par18/fastneg/run/inputs/meta.npz", allow_pickle=False)
    token_rows = np.array([str(t) for t in m["key_token"][A["idx"]]])
    sc = A["g"][..., 2:11]
    base = _auc_point(FL, zL, lab_dac, A["code"])
    curve = []
    for sigma in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        s = plant_scalar(sc, sigma, token_rows)
        zc, _ = L.oof_stack(zL, s.astype(np.float32), lab_dac, A["fold"])
        curve.append({"sigma_m": sigma, "delta_F_ideal": _auc_point(FL, zc, lab_dac, A["code"]) - base, "scalar_alone_F": _auc_point(FL, s.astype(np.float32), lab_dac, A["code"])})
    lo, hi = 0.5, 12.0
    for _ in range(40):                                   # delta_F is decreasing in sigma: bisect to +0.03
        mid = (lo + hi) / 2
        s = plant_scalar(sc, mid, token_rows)
        zc, _ = L.oof_stack(zL, s.astype(np.float32), lab_dac, A["fold"])
        d = _auc_point(FL, zc, lab_dac, A["code"]) - base
        lo, hi = (mid, hi) if d > L.POWER_EFFECT else (lo, mid)
    sigma = (lo + hi) / 2
    s = plant_scalar(sc, sigma, token_rows)
    zc, _ = L.oof_stack(zL, s.astype(np.float32), lab_dac, A["fold"])
    res = {"auc_F_L": base, "curve": curve, "sigma_m": sigma, "delta_F_ideal_at_sigma": _auc_point(FL, zc, lab_dac, A["code"]) - base, "target": L.POWER_EFFECT,
           "scalar_alone_F_at_sigma": _auc_point(FL, s.astype(np.float32), lab_dac, A["code"]),
           "note": "ideal reader = leave-one-fold-out 2-parameter stacker of [z_L, min_k(noisy signed clearance)]; POINT estimates on all 4,941 PRIVGEOM sets; the branch reads the plant through attention and will not be ideal"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("auc_F_L", "sigma_m", "delta_F_ideal_at_sigma", "scalar_alone_F_at_sigma")}))
    print(json.dumps(curve))


def cmd_mde(a):
    FL, PK = L.refs()
    A, lab = load_pg()
    lab_dac = lab[..., L.DAC_COL]
    zL = np.load(f"{PG}/run/fits/L_dac_s0.npy")
    zD = np.load(f"{PG}/run/descr/fits/drop0_dac_s0.npy")                 # DAC-nosign WITHOUT |c_min|: the sign-clean 9-column arm, +0.028 [+0.013, +0.043] registered-recipe linear read
    plan = json.load(open(a.plan, encoding="utf-8"))
    m = np.load("C:/Users/Admin/par18/fastneg/run/inputs/meta.npz", allow_pickle=False)
    kt = np.array([str(t) for t in m["key_token"][A["idx"]]])
    local = np.isin(kt, np.array(plan["tokens"]))
    out = {}
    for name, sel in (("full_4941_sets", np.ones(len(zL), bool)), ("local_ctx_sets", local)):
        li = A["log_idx"][sel]
        u, loc = np.unique(li, return_inverse=True)
        an = L.Analyzer(lab_dac[sel], A["code"][sel], loc, len(u))
        d = an.diff(an.stats(zD[sel]), an.stats(zL[sel]), "Q4")
        se = (d["hi"] - d["lo"]) / (2 * 1.96)

        def phi(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        eff = L.POWER_EFFECT
        p_reach = 1 - phi((L.INFO_MARGIN + 1.96 * se - eff) / se)
        out[name] = {"n_sets": int(sel.sum()), "n_logs": int(len(u)), "delta_F_sign_clean_linear": d, "se_from_interval": se,
                     "MDE_for_CTX_REACHABLE_lower_CI_gt_0.02": L.INFO_MARGIN + 1.96 * se,
                     "P_CTX_REACHABLE_per_seed_if_true_effect_0.03_normal_approx": p_reach,
                     "P_both_seeds_if_independent": p_reach ** 2,
                     "P_NOT_REACHABLE_per_seed_if_true_effect_0_normal_approx": phi((L.NOTREACH_MARGIN - 1.96 * se) / se),
                     "note": "se is read from the banked sign-clean arm's interval (effect +0.028); an interval at a true zero effect is narrower (PRIVGEOM RESULT 5.3: the centred null's interval is ~0.001 wide), so P_NOT is CONSERVATIVE; seeds are floor-of-training-variance only (H-ESTIM-SEED-1)"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_atest(a):
    import sh5_admit as AD
    A, lab = load_pg()
    lab_dac = lab[..., L.DAC_COL]
    g = A["g"]
    cand = np.abs(g[..., 2:11]).astype(np.float64)
    leaky = np.abs(g[..., [0] + list(range(2, 11))]).astype(np.float64)
    res = AD.a_test(cand, lab_dac, leaky)
    if a.mlp:
        res["mlp_recompute_descriptive"] = {"candidate": AD.mlp_recompute_auc(cand.astype(np.float32), lab_dac, A["fold"], A["code"], A["log_idx"], int(A["log_idx"].max()) + 1),
                                           "known_leaky_10col": AD.mlp_recompute_auc(leaky.astype(np.float32), lab_dac, A["fold"], A["code"], A["log_idx"], int(A["log_idx"].max()) + 1)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("admissible", "controls_ok")}))
    if a.mlp:
        print(json.dumps(res["mlp_recompute_descriptive"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plant", "mde", "atest"):
        p = sp.add_parser(n)
        p.add_argument("--out", required=True)
        if n == "mde":
            p.add_argument("--plan", required=True)
        if n == "atest":
            p.add_argument("--mlp", action="store_true")
    a = ap.parse_args()
    {"plant": cmd_plant, "mde": cmd_mde, "atest": cmd_atest}[a.cmd](a)


if __name__ == "__main__":
    main()
