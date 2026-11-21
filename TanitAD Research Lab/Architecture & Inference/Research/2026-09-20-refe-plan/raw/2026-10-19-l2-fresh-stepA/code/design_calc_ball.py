#!/usr/bin/env python3
"""L2-FRESH-STEPA DESIGN CALCULATION for Amendment 2 (MM_RULING_L2_FRESH_D.md): can READ 1 extended to ALL 1,305 B tokens (24 logs) narrow the A1 interval to a half-width <= 0.025?

CPU only.  Reads ONLY data already read: Step A's banked bundle / labels / perts / scores (`D:/Projects/TanitAD-artifacts/l1_stepA_run/`) and L2's banked out-of-fold EP scores
(`l2_offprop/run/arms/H_s{0,1}.npz`, `P2.npz`).  NO carve-B label, score, or feature is read: the only carve-B input is the per-log TOKEN COUNT (`fresh_universe_probe.json` -> `U0_B/B_per_log`), a
number fixed by the plan, not a measurement of anything B scores.  Nothing here touches the chain, the GPU, the sealed `analyse_fresh.py`, or any pinned file.

ESTIMATOR (stated, never bare): A1 = `stepA.analyse(...)["all"]["A1"]`, POOLED lon + lat, a ratio of sums of per-pair concordance over pairs with |true dEP| >= 0.05; the interval is the PAIRED LOG-CLUSTER
BOOTSTRAP (resample the 24 logs, B = 10,000, seed 20260927, `stepA.boot_indices`; percentile 2.5 / 97.5).  It answers "would another draw of LOGS say this" ONLY (H-ESTIM-SEED-1: not another training seed).

STEPS
  (a) IDENTITY: `stepA.analyse` on the banked OOF scores must reproduce the banked L2 A1 (point, lo, hi) EXACTLY (== on the floats) for H_s0 and H_s1 (and P2 as a control); a token-level re-derivation of
      the same interval (per-token num/den -> per-log sums -> `ratio_ci`) must agree to 1e-12.  A control that must DIFFER: one perturbed arm (the EP scores shifted by seed-free rolling of the tokens) must
      NOT reproduce.  Any failure -> exit 3, nothing else is written as a result.
  (b) VARIANCE DECOMPOSITION by SUBSAMPLING tokens within each log at f in {0.25, 0.50, 0.75, 1.00} of Step A's own allocation (R reps each at f < 1; stochastic rounding, >= 1 token per log), the bootstrap
      variance V_f of the ratio for each draw, and the fit   V = a + b / n   (n = mean tokens per log of the draw; a = BETWEEN-LOG floor, b / n = WITHIN-LOG part).  Fit window n in [~4, 16.7] tokens per
      log, n_levels = 4, R reps per level; R^2 reported on (i) the 4 level means and (ii) all draws.  Uncertainty of (a, b) and of the projection: an OUTER bootstrap over the 24 logs (resample logs with
      replacement, redo the whole subsample + fit).  CROSS-CHECKS: (b2) an allocation-exact regression V = alpha*u + beta*w (u = sum k^2/(sum k)^2, w = 1/sum k, k = tokens kept per log), which does not
      assume the allocation is proportional; (b3) a one-way ANOVA method-of-moments decomposition of the linearised per-token residuals e_i = num_i - r*den_i.
  (c) PROJECTION to the B-ALL allocation (1,305 tokens, per-log counts from `B_per_log`).  Three ways, reported side by side: the proportional-scale form V = a + b/n_B; the allocation-exact form
      with the real B-ALL counts; a parametric SIMULATION of the sealed estimator (simulate per-log sums from the fitted model, run the sealed bootstrap) which also checks the mapping at Step A's own
      allocation (must return ~ the observed 0.0452).  A model-free UPPER bracket: resample Step A's tokens WITH replacement to the B-ALL per-log counts (plug-in; it carries Step A's within-log noise as if
      it were signal, so it is an upper bound on the half-width).  The n -> infinity FLOOR (b = 0) is the lower bound of any extrapolation in n.
  EXTRAPOLATION FLAG: B-ALL is 54.4 tokens per log against a fitted maximum of 16.7 = 3.26x, which EXCEEDS the 2x rule; the floor makes the verdict insensitive to it (see the output).
  (d) SIZING (for the branch that is not B-ALL): logs needed for a half-width <= 0.025 as a function of tokens per log, and a consistency check against the earlier READ-2 (universe D) estimate.

    python design_calc_ball.py [--reps 300] [--outer 200] [--outer-reps 24] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
import l2lib  # noqa: E402
import stepA  # noqa: E402

FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
ARMS = "D:/Projects/TanitAD-artifacts/l2_offprop/run/arms"
BANKED = "D:/Projects/TanitAD-artifacts/l2_offprop/run/L2_RESULT.json"
PROBE = "C:/Users/Admin/par18/l2fresh/raw/fresh_universe_probe.json"
OUT_JSON = "C:/Users/Admin/par18/l2fresh/raw/design_calc_ball.json"
HW_TARGET = 0.025                       # the MM ruling's branch threshold
F_LEVELS = (0.25, 0.50, 0.75, 1.00)
CALC_SEED = 20261020                    # RNG for subsampling / outer bootstrap / simulation (NOT the sealed bootstrap seed, which only drives the log resamples)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ data
def load_arms():
    D = l2lib.load_real(FEATS)
    arms = {"P2": (l2lib.ep_apply(D.tok_in, D.W6, D.b6), l2lib.ep_apply(D.tok_p, D.W6, D.b6))}
    for s in (0, 1):
        z = np.load(f"{ARMS}/H_s{s}.npz", allow_pickle=False)
        arms[f"H_s{s}"] = (z["ep_in"], z["ep_p"])
    return D, arms


def token_terms(D, ei, ep):
    """A1's per-pair (num, den) from L1's own `admission_terms`, summed to tokens"""
    P = l2lib._arm_pairs(D, ei, ep)
    num, den = stepA.admission_terms(P)["A1"]
    nt = np.bincount(D.pair_tok, weights=num, minlength=D.N)
    dt = np.bincount(D.pair_tok, weights=den, minlength=D.N)
    return nt, dt


def per_log(nt, dt, D):
    return np.bincount(D.log_idx, weights=nt, minlength=D.n_logs), np.bincount(D.log_idx, weights=dt, minlength=D.n_logs)


# ------------------------------------------------------------------------------------------------------------------------------ (a) identity
def identity_check(D, arms, banked):
    out = {"estimator": "stepA.analyse(...)['all']['A1']: pooled lon+lat ratio of sums, paired log-cluster bootstrap, B=10,000, seed 20260927, percentile 2.5/97.5", "arms": {}}
    ok_all = True
    for name, (ei, ep) in arms.items():
        P = l2lib._arm_pairs(D, ei, ep)
        res = stepA.analyse(P, D.pair_log, D.n_logs, l2lib.BOOT_B, l2lib.BOOT_SEED)["all"]["A1"]
        key = "P2" if name == "P2" else name
        b = banked["arms"][key]["all"]["A1"]
        exact = all(res[k] == b[k] for k in ("point", "lo", "hi", "den"))
        nt, dt = token_terms(D, ei, ep)
        ng, dg = per_log(nt, dt, D)
        ci2 = stepA.ratio_ci(ng, dg, D.boot_idx)
        tok_ok = max(abs(ci2[k] - res[k]) for k in ("point", "lo", "hi")) < 1e-12
        out["arms"][name] = {"recomputed": {k: res[k] for k in ("point", "lo", "hi", "den")}, "banked": {k: b[k] for k in ("point", "lo", "hi", "den")},
                             "bit_exact_vs_banked": bool(exact), "token_level_rederivation_max_abs_diff": max(abs(ci2[k] - res[k]) for k in ("point", "lo", "hi")), "token_level_ok": bool(tok_ok),
                             "rounded_3dp": [round(res["point"], 3), round(res["lo"], 3), round(res["hi"], 3)]}
        ok_all &= bool(exact and tok_ok)
    # the stated headline: 0.827 [0.783, 0.873] (s0)
    h = out["arms"]["H_s0"]["rounded_3dp"]
    out["headline_s0_matches_0.827_0.783_0.873"] = h == [0.827, 0.783, 0.873]
    ok_all &= out["headline_s0_matches_0.827_0.783_0.873"]
    # RED arm: an altered arm must NOT reproduce (identity is not trivially true): roll the s0 EP scores of the anchors by one token
    ei, ep = arms["H_s0"]
    Pr = l2lib._arm_pairs(D, np.roll(ei, 1, axis=0), np.roll(ep, 1, axis=0))
    r = stepA.analyse(Pr, D.pair_log, D.n_logs, l2lib.BOOT_B, l2lib.BOOT_SEED)["all"]["A1"]
    b0 = banked["arms"]["H_s0"]["all"]["A1"]
    out["red_arm_rolled_tokens"] = {"A1": [r["point"], r["lo"], r["hi"]], "differs_from_banked": bool(r["point"] != b0["point"]), "expected": "differs"}
    ok_all &= out["red_arm_rolled_tokens"]["differs_from_banked"]
    out["pass"] = bool(ok_all)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ bootstrap statistics
def boot_stats(num_g, den_g, idx):
    n, d = num_g[idx].sum(1), den_g[idx].sum(1)
    ok = d > 0
    r = n[ok] / d[ok]
    lo, hi = np.quantile(r, [0.025, 0.975])
    return float(r.var(ddof=1)), float((hi - lo) / 2.0), float(num_g.sum() / den_g.sum())


def draw_sums(clusters, nt, dt, f, rng):
    G = len(clusters)
    num, den, K = np.empty(G), np.empty(G), np.empty(G)
    for g, tk in enumerate(clusters):
        if f >= 1.0:
            pick = tk
        else:
            kf = f * len(tk)
            k = int(kf) + int(rng.random() < kf - int(kf))
            k = min(max(k, 1), len(tk))
            pick = rng.choice(tk, k, replace=False)
        num[g], den[g], K[g] = nt[pick].sum(), dt[pick].sum(), len(pick)
    return num, den, K


def run_levels(clusters, nt, dt, idx, reps, rng):
    """-> list of dict(f, n, V, hw, u, w) over all draws (f=1 is deterministic: one draw)"""
    rows = []
    for f in F_LEVELS:
        for _ in range(1 if f >= 1.0 else reps):
            num, den, K = draw_sums(clusters, nt, dt, f, rng)
            V, hw, pt = boot_stats(num, den, idx)
            M = K.sum()
            rows.append({"f": f, "n": float(M / len(clusters)), "V": V, "hw": hw, "u": float((K ** 2).sum() / M ** 2), "w": float(1.0 / M), "pt": pt, "c": hw / (1.96 * np.sqrt(V))})
    return rows


def fit_a_b(x, y):
    """y = a + b x by OLS; R^2"""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return float(beta[0]), float(beta[1]), (1.0 - float((res ** 2).sum()) / sst) if sst > 0 else float("nan")


def fit_alpha_beta(u, w, y):
    """y = alpha*u + beta*w (through the origin, allocation-exact); R^2 about the mean"""
    X = np.column_stack([u, w])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return float(beta[0]), float(beta[1]), (1.0 - float((res ** 2).sum()) / sst) if sst > 0 else float("nan")


def level_fit(rows):
    f = np.array([r["f"] for r in rows])
    n = np.array([r["n"] for r in rows])
    V = np.array([r["V"] for r in rows])
    u = np.array([r["u"] for r in rows])
    w = np.array([r["w"] for r in rows])
    levels = sorted(set(f))
    nm = np.array([n[f == l].mean() for l in levels])
    Vm = np.array([V[f == l].mean() for l in levels])
    a, b, r2_means = fit_a_b(1.0 / nm, Vm)
    a_all, b_all, r2_all = fit_a_b(1.0 / n, V)
    al, be, r2_exact = fit_alpha_beta(u, w, V)
    return {"a": a, "b": b, "R2_level_means": r2_means, "n_levels": len(levels), "level_n": nm.tolist(), "level_V": Vm.tolist(),
            "all_draws": {"a": a_all, "b": b_all, "R2": r2_all, "n_draws": int(len(V))}, "exact": {"alpha": al, "beta": be, "R2": r2_exact}}


# ------------------------------------------------------------------------------------------------------------------------------ projection helpers
def proj_prop(a, b, n_B):
    return a + b / n_B


def proj_exact(alpha, beta, mB):
    M = mB.sum()
    return alpha * (mB ** 2).sum() / M ** 2 + beta / M


def hw_from_V(V, c):
    return 1.96 * c * np.sqrt(max(V, 0.0))


def simulate_hw(alpha, beta, dbar, r, m, idx, rng, reps=300):
    """the SEALED estimator on data simulated from the fitted random-effects model: den_g = dbar*m_g, num_g = r*den_g + E_g, E_g ~ N(0, d^2 (alpha m_g^2 + beta m_g)).  Returns median / 5-95 % of the half-width"""
    hws = []
    for _ in range(reps):
        sd = dbar * np.sqrt(np.maximum(alpha * m ** 2 + beta * m, 0.0))
        E = rng.normal(0.0, 1.0, size=len(m)) * sd
        den = dbar * m
        num = r * den + E
        hws.append(boot_stats(num, den, idx)[1])
    hws = np.array(hws)
    return {"median": float(np.median(hws)), "p05": float(np.quantile(hws, 0.05)), "p95": float(np.quantile(hws, 0.95)), "reps": reps}


def plugin_upper(clusters, nt, dt, mB, idx, rng, reps=300):
    """resample Step A's tokens WITH replacement to the B-ALL per-log counts; the plug-in carries Step A's own within-log noise as if it were signal -> an UPPER bound"""
    hws = []
    for _ in range(reps):
        num, den = np.empty(len(clusters)), np.empty(len(clusters))
        for g, tk in enumerate(clusters):
            pick = rng.choice(tk, int(mB[g]), replace=True)
            num[g], den[g] = nt[pick].sum(), dt[pick].sum()
        hws.append(boot_stats(num, den, idx)[1])
    hws = np.array(hws)
    return {"median": float(np.median(hws)), "p05": float(np.quantile(hws, 0.05)), "p95": float(np.quantile(hws, 0.95)), "reps": reps}


def anova_decomp(nt, dt, D):
    """one-way random-effects method of moments on e_i = num_i - r*den_i (r = pooled ratio).  sigma_w^2 = pooled within-log variance of e; sigma_b^2 from sum_g E_g^2 = sum(m^2 sb2 + m sw2)"""
    r = nt.sum() / dt.sum()
    e = nt - r * dt
    m = np.bincount(D.log_idx, minlength=D.n_logs).astype(float)
    E = np.bincount(D.log_idx, weights=e, minlength=D.n_logs)
    ebar = E / m
    sw2 = float(sum(((e[D.log_idx == g] - ebar[g]) ** 2).sum() for g in range(D.n_logs)) / (D.N - D.n_logs))
    # E_g is centred (sum_g E_g = 0), so E[sum E_g^2] = sum m^2 sb2 + sum m sw2 - (sum_g ... centring term); the first-order correction is the factor (G-1)/G on sum m^2 sb2 + sum m sw2
    G = D.n_logs
    sb2 = float((((E ** 2).sum() * G / (G - 1)) - sw2 * m.sum()) / (m ** 2).sum())
    dbar = float(dt.mean())
    return {"r": float(r), "dbar": dbar, "sigma_w2": sw2, "sigma_b2": sb2, "ICC_of_e": sb2 / (sb2 + sw2), "alpha": sb2 / dbar ** 2, "beta": sw2 / dbar ** 2,
            "note": "sigma_b2 may be negative at 24 logs (method of moments); if so the floor is 0 on this estimator and the fit is not informative about it"}


# ------------------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--outer", type=int, default=200)
    ap.add_argument("--outer-reps", type=int, default=24)
    ap.add_argument("--sim-reps", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.outer, a.outer_reps, a.sim_reps = 10, 6, 4, 20
    t0 = time.time()
    banked = json.load(open(BANKED))
    D, arms = load_arms()
    probe = json.load(open(PROBE))["U0_B"]
    logs = sorted(probe["B_per_log"])
    if logs != D.log_names:
        raise SystemExit("B_per_log names differ from Step A's 24 logs")
    mB = np.array([probe["B_per_log"][l] for l in logs], dtype=float)
    mA = np.bincount(D.log_idx).astype(float)
    result = {"what": "L2-FRESH-STEPA design calculation for Amendment 2: projected A1 half-width at B-ALL (1,305 tokens, 24 logs)", "evidence": "ESTIMATED (a model fitted on Step A's own banked per-token terms; MEASURED inputs, projected output)",
              "estimator": "paired log-cluster bootstrap (24 logs, 10,000, seed 20260927), percentile 2.5/97.5; answers 'another draw of LOGS' only (H-ESTIM-SEED-1)",
              "inputs": {"feats": FEATS, "arms_dir": ARMS, "banked": BANKED, "banked_sha256": sha256_file(BANKED), "probe": PROBE, "probe_sha256": sha256_file(PROBE), "carve_B_labels_or_scores_read": False},
              "calc_seed": CALC_SEED, "args": vars(a)}
    idn = identity_check(D, arms, banked)
    result["identity"] = idn
    print("IDENTITY", json.dumps({k: v["recomputed"] for k, v in idn["arms"].items()}), "pass=", idn["pass"], flush=True)
    if not idn["pass"]:
        json.dump(l2lib.jsonable(result), open(OUT_JSON, "w"), indent=1)
        print("IDENTITY_FAIL: stop and report", flush=True)
        sys.exit(3)
    result["context_banked_A4_A5_s0"] = {"A1": banked["arms"]["H_s0"]["all"]["A1"], "A4": banked["arms"]["H_s0"]["all"]["A4"], "A5": banked["arms"]["H_s0"]["all"]["A5"],
                                          "note": "A4's interval (gate <= 0.10) and A5's point (gate <= 0.05) are nowhere near their gates, so A1's interval is the term that decides O1 vs O3 under interval_aware; this calc projects A1 ONLY"}
    result["allocation"] = {"stepA_tokens": int(mA.sum()), "B_ALL_tokens": int(mB.sum()), "logs": int(D.n_logs), "stepA_per_log": mA.astype(int).tolist(), "B_ALL_per_log": mB.astype(int).tolist(),
                            "corr_per_log_counts": float(np.corrcoef(mA, mB)[0, 1]), "ratio_B_over_A_per_log_min_max": [float((mB / mA).min()), float((mB / mA).max())],
                            "G_eff_stepA": float(mA.sum() ** 2 / (mA ** 2).sum()), "G_eff_B_ALL": float(mB.sum() ** 2 / (mB ** 2).sum()),
                            "n_per_log_stepA": float(mA.mean()), "n_per_log_B_ALL": float(mB.mean()), "extrapolation_factor_in_n": float(mB.mean() / mA.mean()),
                            "extrapolation_flag": "B-ALL tokens per log = %.2fx the fitted maximum (Step A, f=1): BEYOND the 2x rule" % (mB.mean() / mA.mean())}
    clusters = [np.where(D.log_idx == g)[0] for g in range(D.n_logs)]
    result["arms"] = {}
    for name in ("H_s0", "H_s1", "P2"):
        ei, ep = arms[name]
        nt, dt = token_terms(D, ei, ep)
        dbar = float(dt.mean())
        rng = np.random.default_rng(CALC_SEED + {"H_s0": 0, "H_s1": 1, "P2": 2}[name])
        t1 = time.time()
        rows = run_levels(clusters, nt, dt, D.boot_idx, a.reps, rng)
        fit = level_fit(rows)
        c_all = float(np.mean([r["c"] for r in rows]))
        c_obs = [r["c"] for r in rows if r["f"] == 1.0][0]
        V_obs = [r["V"] for r in rows if r["f"] == 1.0][0]
        hw_obs = [r["hw"] for r in rows if r["f"] == 1.0][0]
        # ---- projection, three ways
        nB = float(mB.mean())
        V_prop = proj_prop(fit["a"], fit["b"], nB)
        V_exact = proj_exact(fit["exact"]["alpha"], fit["exact"]["beta"], mB)
        V_floor = fit["a"]
        n2x = 2.0 * float(mA.mean())
        V_2x = proj_prop(fit["a"], fit["b"], n2x)
        proj = {"c_bootstrap_to_normal_mean_over_draws": c_all, "c_at_f1": c_obs, "V_obs_f1": V_obs, "hw_obs_f1": hw_obs,
                "proportional_form": {"V": V_prop, "HW": hw_from_V(V_prop, c_all)},
                "allocation_exact_form": {"V": V_exact, "HW": hw_from_V(V_exact, c_all)},
                "floor_n_to_infinity_(b=0)": {"V": V_floor, "HW": hw_from_V(V_floor, c_all)},
                "at_2x_fitted_range_n=%.1f_(the_admissible_extrapolation_limit)" % n2x: {"V": V_2x, "HW": hw_from_V(V_2x, c_all)},
                "stepA_f1_check_HW": hw_from_V(proj_prop(fit["a"], fit["b"], float(mA.mean())), c_all)}
        # ---- the sealed estimator on simulated data: mapping check at Step A's allocation, then B-ALL
        al, be = fit["exact"]["alpha"], fit["exact"]["beta"]
        r_pt = float(nt.sum() / dt.sum())
        sim_A = simulate_hw(al, be, dbar, r_pt, mA, D.boot_idx, np.random.default_rng(CALC_SEED + 10), a.sim_reps)
        sim_B = simulate_hw(al, be, dbar, r_pt, mB, D.boot_idx, np.random.default_rng(CALC_SEED + 11), a.sim_reps)
        proj["simulated_sealed_estimator"] = {"stepA_allocation_(must_be_near_observed_%.4f)" % hw_obs: sim_A, "B_ALL_allocation": sim_B}
        proj["plugin_resample_UPPER_bracket"] = plugin_upper(clusters, nt, dt, mB, D.boot_idx, np.random.default_rng(CALC_SEED + 12), a.sim_reps)
        # ---- ANOVA cross-check
        an = anova_decomp(nt, dt, D)
        V_an_A = (an["sigma_b2"] * (mA ** 2).sum() + an["sigma_w2"] * mA.sum()) / (an["dbar"] * mA.sum()) ** 2
        V_an_B = (an["sigma_b2"] * (mB ** 2).sum() + an["sigma_w2"] * mB.sum()) / (an["dbar"] * mB.sum()) ** 2
        an["V_model_at_stepA"], an["V_model_at_B_ALL"] = float(V_an_A), float(V_an_B)
        an["HW_at_stepA_(vs_observed_%.4f)" % hw_obs], an["HW_at_B_ALL"] = hw_from_V(V_an_A, c_all), hw_from_V(V_an_B, c_all)
        # closed-form between-log FLOOR at the B-ALL allocation (tokens per log -> infinity): HW = 1.96 c sqrt(sigma_b^2 / (dbar^2 G_eff)); a normal-theory 95 % LOWER bound on sigma_b^2 from the chi-square with
        # df = G_eff - 1 (an approximation: log effects are not normal and G_eff is only ~12.6); reported beside the outer bootstrap, which makes no normality assumption
        from scipy.stats import chi2
        geff_B = float(mB.sum() ** 2 / (mB ** 2).sum())
        df = geff_B - 1.0
        sb2 = max(an["sigma_b2"], 0.0)
        floor_cf = hw_from_V(sb2 / (an["dbar"] ** 2 * geff_B), c_all)
        sb2_lo, sb2_hi = sb2 * df / chi2.ppf(0.975, df), sb2 * df / chi2.ppf(0.025, df)
        an["floor_closed_form_B_ALL"] = {"G_eff": geff_B, "df": df, "HW_floor_point": floor_cf, "sigma_b2_95pct_normal_theory": [sb2_lo, sb2_hi],
                                         "HW_floor_95pct_normal_theory": [hw_from_V(sb2_lo / (an["dbar"] ** 2 * geff_B), c_all), hw_from_V(sb2_hi / (an["dbar"] ** 2 * geff_B), c_all)],
                                         "sigma_b2_needed_for_floor_HW<=0.025": float((HW_TARGET / (1.96 * c_all)) ** 2 * an["dbar"] ** 2 * geff_B)}
        # ---- the sizing table (a balanced new data job: G' logs x n tokens per log)
        sizing = {}
        for n in (1, 2, 4, 8, 16.7, 54.4, 1e9):
            Vn = (al + be / n)
            G_need = Vn * (1.96 * c_all) ** 2 / HW_TARGET ** 2           # V = (alpha + beta/n) / G'
            sizing["n=%g" % n] = {"logs_needed_for_HW<=0.025": float(np.ceil(G_need)), "tokens_total": float(np.ceil(G_need) * min(n, 1e9)) if n < 1e8 else None}
        dcheck = (al / 84.0 + be / 99.0)
        proj["sizing"] = sizing
        earlier = json.load(open("C:/Users/Admin/par18/l2fresh/raw/readtwo_geometry.json"))["arms"].get("L2_OOF_" + name[2:] if name != "P2" else "P2", {}).get("HW_D_estimated")
        proj["READ2_universe_D_consistency_check"] = {"HW_via_this_model_(84_logs,_99_tokens,_balanced)": hw_from_V(dcheck, c_all), "HW_via_the_earlier_DEFF_estimate_(readtwo_geometry.json)": earlier}
        # ---- the outer bootstrap over logs: the uncertainty of the fit and of the projection
        outer = {"a": [], "b": [], "HW_prop": [], "HW_exact": [], "HW_floor": []}
        rng_o = np.random.default_rng(CALC_SEED + 100 + {"H_s0": 0, "H_s1": 1, "P2": 2}[name])
        for _ in range(a.outer):
            j = rng_o.integers(0, D.n_logs, D.n_logs)
            cl = [clusters[g] for g in j]
            rows_b = run_levels(cl, nt, dt, D.boot_idx, a.outer_reps, rng_o)
            fb = level_fit(rows_b)
            cb = float(np.mean([r["c"] for r in rows_b]))
            mBb, mAb = mB[j], mA[j]
            outer["a"].append(fb["a"])
            outer["b"].append(fb["b"])
            outer["HW_prop"].append(hw_from_V(proj_prop(fb["a"], fb["b"], float(mB.mean())), cb))
            outer["HW_exact"].append(hw_from_V(proj_exact(fb["exact"]["alpha"], fb["exact"]["beta"], mBb), cb))
            outer["HW_floor"].append(hw_from_V(fb["a"], cb))
        q = lambda v: [float(np.quantile(v, 0.025)), float(np.median(v)), float(np.quantile(v, 0.975))]
        proj["outer_bootstrap_over_logs"] = {"n_outer": a.outer, "reps_per_level": a.outer_reps, "a_2.5_50_97.5": q(outer["a"]), "b_2.5_50_97.5": q(outer["b"]),
                                             "HW_proportional_2.5_50_97.5": q(outer["HW_prop"]), "HW_allocation_exact_2.5_50_97.5": q(outer["HW_exact"]), "HW_floor_2.5_50_97.5": q(outer["HW_floor"]),
                                             "share_of_outer_draws_with_HW_proportional_<=0.025": float(np.mean(np.array(outer["HW_prop"]) <= HW_TARGET)),
                                             "share_of_outer_draws_with_HW_floor_<=0.025": float(np.mean(np.array(outer["HW_floor"]) <= HW_TARGET)),
                                             "note": "the outer bootstrap re-draws 24 logs with replacement; the 24-log fit is itself noisy, this is the honest width of the projection"}
        result["arms"][name] = {"A1_point": r_pt, "dbar_pairs_per_token": dbar, "fit_window_n_tokens_per_log": [fit["level_n"][0], fit["level_n"][-1]], "fit": fit, "projection": proj, "anova_crosscheck": an,
                                "seconds": time.time() - t1}
        print(f"ARM {name} A1={r_pt:.4f} obs HW(f=1)={hw_obs:.4f}  fit a={fit['a']:.3e} b={fit['b']:.3e} R2(means,n=4)={fit['R2_level_means']:.3f} R2(all {fit['all_draws']['n_draws']} draws)={fit['all_draws']['R2']:.3f} "
              f" HW_B-ALL prop={proj['proportional_form']['HW']:.4f} exact={proj['allocation_exact_form']['HW']:.4f} floor={proj['floor_n_to_infinity_(b=0)']['HW']:.4f} "
              f" sim={sim_B['median']:.4f} plugin_upper={proj['plugin_resample_UPPER_bracket']['median']:.4f} outer95(prop)={proj['outer_bootstrap_over_logs']['HW_proportional_2.5_50_97.5']}", flush=True)
    # ---- the branch (pre-stated in Amendment 2): the LARGER of the two seeds' projections is compared with 0.025
    hw_seed = {s: result["arms"][s]["projection"]["proportional_form"]["HW"] for s in ("H_s0", "H_s1")}
    hw_floor = {s: result["arms"][s]["projection"]["floor_n_to_infinity_(b=0)"]["HW"] for s in ("H_s0", "H_s1")}
    hw_lo_outer = {s: result["arms"][s]["projection"]["outer_bootstrap_over_logs"]["HW_floor_2.5_50_97.5"][0] for s in ("H_s0", "H_s1")}
    result["branch"] = {"threshold": HW_TARGET, "projected_HW_by_seed_(proportional_form)": hw_seed, "floor_HW_by_seed": hw_floor, "floor_HW_outer_2.5pct_by_seed": hw_lo_outer,
                        "selected": "B-ALL" if max(hw_seed.values()) <= HW_TARGET else "UNSEEN-STRATUM DATA JOB",
                        "robust_to_the_extrapolation_flag": bool(min(hw_floor.values()) > HW_TARGET),
                        "robust_to_the_fit_noise_(outer_2.5pct_of_the_floor_above_threshold)": bool(min(hw_lo_outer.values()) > HW_TARGET)}
    result["seconds_total"] = time.time() - t0
    json.dump(l2lib.jsonable(result), open(OUT_JSON, "w"), indent=1)
    print("BRANCH", json.dumps(result["branch"]), flush=True)
    print("DESIGN_CALC_DONE", OUT_JSON, "%.0fs" % result["seconds_total"], flush=True)


if __name__ == "__main__":
    main()
