"""Amendment 2 power calculation (CPU, numpy).  Reads PRIVGEOM-SAFE's banked arrays READ-ONLY, exactly as `sh5_design_calc.py mde` does, and produces NO SH5 outcome (no SH5 number exists).

    python sh5_amend2_power.py --plan work/plan/plan.json --mde work/mde.json --falsefire work/falsefire_sim.json --out work/amend2_power.json

It answers, BEFORE Amendment 2 is sealed (MM ruling `MM_RULING_SH5.md` s1):
  * the per-seed and both-seed P(CTX-REACHABLE | true Delta_F) under the SEALED rule (lower CI > +0.02) and the NEW rule (lower CI > 0 AND point >= +0.02), LOCAL and FULL, with the SAME standard errors that gave the
    sealed MDEs (+0.050 LOCAL, +0.035 FULL): the half-width of the banked sign-clean arm's interval, `se = (hi - lo) / (2 * 1.96)`.  Reproduced here from the arrays and checked against work/mde.json.
  * P(false CTX-REACHABLE | true Delta_F = 0) under both rules.
  * a NESTED-BOOTSTRAP cross-check that does not assume the interval is normal or symmetric (an outer bootstrap over the analysis set's logs, an inner percentile bootstrap per outer draw = the registered estimator).
  * the banked CPU false-fire replicates (`falsefire_sim.json`) re-read under the NEW rule.

BETWEEN-SEED CORRELATION (stated, not measured: there is no banked pair of seeds of the sign-clean arm).  Convention = the sealed table's: each seed's POINT estimate has marginal sd = se (the sampling sd of one draw of logs) and the
two seeds' points are bivariate normal with correlation rho; the two seeds share the logs and the bootstrap draws, so rho = 1 (identical) is the upper bound of the both-seed probability and rho = 0 (the sealed table's
"if independent") the lower bound; rho 0.5 and 0.8 are the in-between rows.  SENSITIVITY (worst reasonable case, H-ESTIM-SEED-1): training variance EQUAL to the sampling variance on top of it (marginal sd = sqrt(2) se, rho = 0.5,
the interval still se wide because the bootstrap holds the models fixed).
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
import sh5_design_calc as DC                                                                           # noqa: E402

Z = 1.96                                   # the sealed MDE arithmetic uses 1.96 (sh5_design_calc.py cmd_mde)
SEALED_LO_MIN = L.INFO_MARGIN            # sealed: lower CI > +0.02
NEW_LO_MIN, NEW_POINT_MIN = 0.0, 0.02   # Amendment 2: lower CI > 0 AND point >= +0.02
EFFECTS = (0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06)


def Q(x):
    """upper tail of the standard normal"""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def bvn_both_above(a: float, rho: float) -> float:
    """P(X > a, Y > a) for a standard bivariate normal with correlation rho (deterministic quadrature)."""
    if rho >= 1.0 - 1e-12:
        return Q(a)
    if rho <= 1e-12:
        return Q(a) ** 2
    x = np.linspace(a, max(a, -8.0) + 14.0, 400001)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    s = math.sqrt(1 - rho * rho)
    tail = 0.5 * np.vectorize(math.erfc)(((a - rho * x) / s) / math.sqrt(2.0))
    return float(np.trapz(phi * tail, x))


def thresholds(se: float) -> dict:
    """the minimum POINT estimate that passes each rule, when lower CI = point - 1.96 se (the sealed MDE arithmetic)."""
    t_sealed = SEALED_LO_MIN + Z * se
    lo_cond = NEW_LO_MIN + Z * se
    t_new = max(lo_cond, NEW_POINT_MIN)
    return {"sealed_point_needed": t_sealed, "new_lower_ci_conjunct_needs_point": lo_cond, "new_point_conjunct_needs_point": NEW_POINT_MIN, "new_point_needed": t_new,
            "binding_conjunct_of_new_rule": "lower CI > 0" if lo_cond > NEW_POINT_MIN else "point >= +0.02"}


def p_pass(eff: float, se: float, t: float, rho: float, sd_total: float = None) -> dict:
    """per-seed and both-seed probability that point >= t when point ~ N(eff, sd_total^2) (sd_total defaults to se)"""
    sd = se if sd_total is None else sd_total
    a = (t - eff) / sd
    return {"per_seed": Q(a), "both_seeds": bvn_both_above(a, rho)}


def analytic_table(se: float, rhos=(1.0, 0.8, 0.5, 0.0)) -> dict:
    th = thresholds(se)
    out = {"se": se, "thresholds": th, "by_effect": {}}
    for eff in EFFECTS:
        row = {}
        for name, t in (("sealed", th["sealed_point_needed"]), ("new", th["new_point_needed"])):
            row[name] = {"per_seed": Q((t - eff) / se), "both_seeds_by_rho": {f"rho={r}": bvn_both_above((t - eff) / se, r) for r in rhos}}
        out["by_effect"][f"{eff:+.3f}"] = row
    # the effect at which a seed passes with 80 % probability (the rule's MDE at 80 % power)
    z80 = 0.8416212335729143
    out["MDE_80pct_power_per_seed"] = {"sealed": th["sealed_point_needed"] + z80 * se, "new": th["new_point_needed"] + z80 * se}
    out["MDE_50pct_power_per_seed"] = {"sealed": th["sealed_point_needed"], "new": th["new_point_needed"]}
    # sensitivity: training variance equal to the sampling variance on top of it
    sd2 = math.sqrt(2.0) * se
    sens = {}
    for eff in (0.0, 0.03):
        row = {}
        for name, t in (("sealed", th["sealed_point_needed"]), ("new", th["new_point_needed"])):
            row[name] = {"per_seed": Q((t - eff) / sd2), "both_seeds_rho_0.5": bvn_both_above((t - eff) / sd2, 0.5)}
        sens[f"{eff:+.3f}"] = row
    out["sensitivity_training_var_equals_sampling_var"] = {"marginal_sd": sd2, "rho": 0.5, "by_effect": sens}
    return out


def false_fire_table(se: float, sds=(0.001, 0.002, 0.004)) -> dict:
    """P(false CTX-REACHABLE | true Delta_F = 0).  Row 'sampling_only_se' uses the banked arm's se as the point's sd (CONSERVATIVE: a true-zero effect has a far narrower interval, PRIVGEOM RESULT 5.3 ~0.001 wide, and the
    16 noise-trained CPU replicates scattered with sd 0.0006); the bracketing rows use the stated sd for the point with the interval half-width 1.96 * min(se, sd)."""
    th = thresholds(se)
    out = {}
    for label, sd_pt, se_ci in [("sampling_only_se_of_the_banked_arm", se, se)] + [(f"point_sd_{s}_interval_se_{min(s, se)}", s, min(s, se)) for s in sds] + [("training_var_equals_sampling_var", math.sqrt(2.0) * se, se)]:
        t_s = SEALED_LO_MIN + Z * se_ci
        t_n = max(NEW_LO_MIN + Z * se_ci, NEW_POINT_MIN)
        row = {"point_sd": sd_pt, "ci_se": se_ci}
        for name, t in (("sealed", t_s), ("new", t_n)):
            a = t / sd_pt
            row[name] = {"per_seed": Q(a), "both_seeds_rho=1": Q(a), "both_seeds_rho=0.5": bvn_both_above(a, 0.5), "both_seeds_rho=0": Q(a) ** 2}
        out[label] = row
    return out


# ---- nested bootstrap (outer: draws of LOGS, inner: the registered percentile bootstrap) -----------------------------------------------------------------------------------------------------
def nested(nA, dA, nB, dB, R: int, Bin: int, seed: int) -> dict:
    n = len(nA)
    rng = np.random.default_rng(seed)
    outer = rng.integers(0, n, size=(R, n))
    inner = rng.integers(0, n, size=(Bin, n))
    pt, lo, hi = np.empty(R), np.empty(R), np.empty(R)
    for r in range(R):
        i = outer[r]
        a_n, a_d, b_n, b_d = nA[i], dA[i], nB[i], dB[i]
        pt[r] = a_n.sum() / a_d.sum() - b_n.sum() / b_d.sum()
        v = a_n[inner].sum(1) / a_d[inner].sum(1) - b_n[inner].sum(1) / b_d[inner].sum(1)
        lo[r], hi[r] = np.percentile(v, 2.5), np.percentile(v, 97.5)
    return {"point": pt, "lo": lo, "hi": hi}


def rules(pt, lo, hi):
    sealed = lo > SEALED_LO_MIN
    new = (lo > NEW_LO_MIN) & (pt >= NEW_POINT_MIN)
    return sealed, new


def nested_summary(sim: dict, banked_point: float, effects=("banked", 0.03, 0.0)) -> dict:
    out = {}
    for e in effects:
        shift = 0.0 if e == "banked" else float(e) - banked_point
        pt, lo, hi = sim["point"] + shift, sim["lo"] + shift, sim["hi"] + shift
        s, n = rules(pt, lo, hi)
        row = {"shift_applied": shift, "mean_point": float(pt.mean()), "sd_point": float(pt.std()), "mean_half_width": float(((hi - lo) / 2).mean()),
               "P_sealed_per_seed": float(s.mean()), "P_new_per_seed": float(n.mean()),
               "P_new_lower_ci_gt_0": float((lo > 0).mean()), "P_new_point_ge_0.02": float((pt >= 0.02).mean())}
        # both seeds: the two seeds share the outer draw of logs; training noise u_s ~ N(0, sd_u) per seed shifts that seed's whole interval (the bootstrap holds the models fixed)
        rng = np.random.default_rng(20261020)
        M = 40
        for tag, sd_u in (("sd_u=0 (rho=1)", 0.0), ("sd_u=0.5se (rho=0.8)", 0.5), ("sd_u=1se (rho=0.5)", 1.0)):
            u = rng.normal(0.0, sd_u * float((hi - lo).mean() / (2 * Z)), size=(len(pt), M, 2))
            ps, pn = [], []
            for k in range(2):
                s_k, n_k = rules(pt[:, None] + u[..., k], lo[:, None] + u[..., k], hi[:, None] + u[..., k])
                ps.append(s_k)
                pn.append(n_k)
            row[f"both_seeds[{tag}]"] = {"sealed": float((ps[0] & ps[1]).mean()), "new": float((pn[0] & pn[1]).mean()), "per_seed_sealed": float(ps[0].mean()), "per_seed_new": float(pn[0].mean())}
        out["effect_" + (e if isinstance(e, str) else f"{e:+.3f}")] = row
    return out


def reread_falsefire(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))

    def fires(ci):
        return bool(ci["lo"] > NEW_LO_MIN and ci["point"] >= NEW_POINT_MIN)

    def sealed(ci):
        return bool(ci["lo"] > SEALED_LO_MIN)
    nl = [r for r in d["rows"]]
    pos = d["pos_rows"]
    return {"noise_trained_replicates": {"n": len(nl), "new_rule_fires_F": sum(fires(r["delta_F"]) for r in nl), "sealed_rule_fires_F": sum(sealed(r["delta_F"]) for r in nl),
                                         "new_rule_fires_Q123": sum(fires(r["delta_Q123"]) for r in nl), "max_point_F": max(r["delta_F"]["point"] for r in nl),
                                         "n_lower_ci_gt_0_F": sum(r["delta_F"]["lo"] > 0 for r in nl)},
            "positive_control_replicates": {"n": len(pos), "new_rule_fires_F": sum(fires(r["delta_F"]) for r in pos), "sealed_rule_fires_F": sum(sealed(r["delta_F"]) for r in pos),
                                            "points_F": [r["delta_F"]["point"] for r in pos]},
            "note": "CPU synthetic world (sh5_sim_falsefire.py); 16 noise-trained replicates + 4 branch-reads-the-ctx replicates, re-read under the NEW rule. Valid for a bound on the machinery's false-fire rate, not for the real rig's training variance."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mde", required=True)
    ap.add_argument("--falsefire", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--R", type=int, default=3000)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    FL, PK = L.refs()
    A, lab = DC.load_pg()
    lab_dac = lab[..., L.DAC_COL]
    zL = np.load(f"{DC.PG}/run/fits/L_dac_s0.npy")
    zD = np.load(f"{DC.PG}/run/descr/fits/drop0_dac_s0.npy")
    plan = json.load(open(a.plan, encoding="utf-8"))
    m = np.load("C:/Users/Admin/par18/fastneg/run/inputs/meta.npz", allow_pickle=False)
    kt = np.array([str(t) for t in m["key_token"][A["idx"]]])
    local = np.isin(kt, np.array(plan["tokens"]))
    sealed_mde = json.load(open(a.mde, encoding="utf-8"))
    out = {"purpose": "Amendment 2 power: CTX-REACHABLE sealed rule (lower CI > +0.02) vs NEW rule (lower CI > 0 AND point >= +0.02), on the inputs that gave the sealed MDEs", "evidence_class": "MEASURED inputs (PRIVGEOM banked arrays), ESTIMATED probabilities (normal approximation + nested bootstrap)",
           "inputs": {"privgeom_arrays": f"{DC.PG}/run/join/arrays.npz", "L": f"{DC.PG}/run/fits/L_dac_s0.npy", "sign_clean_arm": f"{DC.PG}/run/descr/fits/drop0_dac_s0.npy", "plan": a.plan, "sealed_mde": a.mde},
           "rule_constants": {"sealed": {"lower_ci_gt": SEALED_LO_MIN}, "new": {"lower_ci_gt": NEW_LO_MIN, "point_ge": NEW_POINT_MIN}, "true_effect_sized_for": L.POWER_EFFECT},
           "correlation_assumption": ("each seed's point estimate has marginal sd = se; the seeds share the logs and the bootstrap draws, so the both-seed probability lies between rho = 1 (identical: = per-seed) and rho = 0 (independent: the sealed table's "
                                      "'both seeds if independent'); rho 0.8 and 0.5 are intermediate. NOT measured (no banked pair of seeds of the sign-clean arm). Sensitivity: training variance equal to sampling variance on top (marginal sd sqrt(2) se, rho 0.5)."),
           "variants": {}}
    for name, key, sel in (("FULL", "full_4941_sets", np.ones(len(zL), bool)), ("LOCAL", "local_ctx_sets", local)):
        li = A["log_idx"][sel]
        u, loc = np.unique(li, return_inverse=True)
        an = L.Analyzer(lab_dac[sel], A["code"][sel], loc, len(u))
        stD, stL = an.stats(zD[sel]), an.stats(zL[sel])
        d = an.diff(stD, stL, "Q4")
        se = (d["hi"] - d["lo"]) / (2 * Z)
        sealed_se = sealed_mde[key]["se_from_interval"]
        assert abs(se - sealed_se) < 1e-12, f"{name}: se {se} does not reproduce the sealed MDE input {sealed_se}"
        assert abs(d["point"] - sealed_mde[key]["delta_F_sign_clean_linear"]["point"]) < 1e-12
        nA, dA = stD["Q4"]
        nB, dB = stL["Q4"]
        sim = nested(nA, dA, nB, dB, a.R, a.B, 20261019)
        tab = analytic_table(se)
        sealed_reproduced = {"MDE_sealed_rule_from_this_run": tab["thresholds"]["sealed_point_needed"], "MDE_sealed_in_mde_json": sealed_mde[key]["MDE_for_CTX_REACHABLE_lower_CI_gt_0.02"],
                             "P_sealed_per_seed_at_0.03_from_this_run": tab["by_effect"]["+0.030"]["sealed"]["per_seed"], "P_sealed_per_seed_at_0.03_in_mde_json": sealed_mde[key]["P_CTX_REACHABLE_per_seed_if_true_effect_0.03_normal_approx"],
                             "P_sealed_both_independent_from_this_run": tab["by_effect"]["+0.030"]["sealed"]["both_seeds_by_rho"]["rho=0.0"], "P_sealed_both_independent_in_mde_json": sealed_mde[key]["P_both_seeds_if_independent"]}
        out["variants"][name] = {"n_sets": int(sel.sum()), "n_logs": int(len(u)), "banked_sign_clean_arm_delta_F": d, "reproduces_sealed_MDE_inputs": sealed_reproduced, "analytic": tab,
                                 "false_fire_true_zero": false_fire_table(se), "nested_bootstrap": {"R_outer": a.R, "B_inner": a.B, "note": "outer = draws of logs from the analysis set, inner = the registered percentile bootstrap; shift = relocation of (point, lo, hi) to the stated true effect at the banked se (conservative for the true zero)",
                                                                                                    "result": nested_summary(sim, d["point"])}}
    out["falsefire_sim_banked_rows_reread_under_new_rule"] = reread_falsefire(a.falsefire)
    head = {}
    for name, v in out["variants"].items():
        t3, t0 = v["analytic"]["by_effect"]["+0.030"], v["analytic"]["by_effect"]["+0.000"]
        n3, n0 = v["nested_bootstrap"]["result"]["effect_+0.030"], v["nested_bootstrap"]["result"]["effect_+0.000"]
        sens = v["analytic"]["sensitivity_training_var_equals_sampling_var"]["by_effect"]
        head[name] = {"se": v["analytic"]["se"], "n_logs": v["n_logs"], "binding_conjunct_of_new_rule": v["analytic"]["thresholds"]["binding_conjunct_of_new_rule"],
                      "MDE_50pct_per_seed": v["analytic"]["MDE_50pct_power_per_seed"], "MDE_80pct_per_seed": v["analytic"]["MDE_80pct_power_per_seed"],
                      "P_CTX_REACHABLE_at_true_+0.03": {r: {"per_seed": t3[r]["per_seed"], "both_seeds": {"rho=1 (shared draw)": t3[r]["both_seeds_by_rho"]["rho=1.0"], "rho=0.8": t3[r]["both_seeds_by_rho"]["rho=0.8"],
                                                            "rho=0.5": t3[r]["both_seeds_by_rho"]["rho=0.5"], "rho=0 (independent)": t3[r]["both_seeds_by_rho"]["rho=0.0"]}} for r in ("sealed", "new")},
                      "P_false_CTX_REACHABLE_at_true_0_CONSERVATIVE_se": {r: {"per_seed": t0[r]["per_seed"], "both_seeds": {"rho=1": t0[r]["both_seeds_by_rho"]["rho=1.0"], "rho=0.5": t0[r]["both_seeds_by_rho"]["rho=0.5"], "rho=0": t0[r]["both_seeds_by_rho"]["rho=0.0"]}} for r in ("sealed", "new")},
                      "nested_bootstrap_per_seed_at_+0.03": {"sealed": n3["P_sealed_per_seed"], "new": n3["P_new_per_seed"], "both_seeds_rho=0.5_equiv": {"sealed": n3["both_seeds[sd_u=1se (rho=0.5)]"]["sealed"], "new": n3["both_seeds[sd_u=1se (rho=0.5)]"]["new"]}},
                      "nested_bootstrap_per_seed_at_true_0": {"sealed": n0["P_sealed_per_seed"], "new": n0["P_new_per_seed"]},
                      "sensitivity_training_var_equals_sampling_var": {"P_at_+0.03": sens["+0.030"], "P_false_at_0": sens["+0.000"]}}
    for name, v in out["variants"].items():                                       # VOID (C-NULL-LABEL, 3 noise-trained replicates, each ONE seed's interval): the sealed fire rule vs the NEW rule
        t0 = v["analytic"]["by_effect"]["+0.000"]
        s0 = v["analytic"]["sensitivity_training_var_equals_sampling_var"]["by_effect"]["+0.000"]
        blk = {}
        for lab, (ps, pn) in (("sampling_only_conservative_se", (t0["sealed"]["per_seed"], t0["new"]["per_seed"])), ("training_var_equals_sampling_var", (s0["sealed"]["per_seed"], s0["new"]["per_seed"]))):
            blk[lab] = {"per_replicate_fires_sealed_rule": ps, "per_replicate_would_fire_under_new_rule": pn, "P_any_of_3_sealed_rule(bounds: all replicates identical .. independent)": [ps, 1 - (1 - ps) ** 3],
                        "P_any_of_3_if_VOID_followed_the_new_rule(bounds)": [pn, 1 - (1 - pn) ** 3]}
        v["void_sensitivity"] = {"note": ("VOID keeps its SEALED numeric rule (lo > +0.02; MM ruling s1). A noise-trained replicate that satisfies the NEW CTX-REACHABLE rule but not the sealed numeric one does not void the run. "
                                          "The alternative (VOID follows the new rule) would void a CLEAN machinery with the 'if VOID followed the new rule' probability: it is the false-alarm cost of that alternative. Replicates share the logs, so the truth lies between the bounds."), **blk}
        head[name]["void_sensitivity"] = v["void_sensitivity"]
    out["headline"] = head
    out["power_script_sha256"] = L.sha256_file(os.path.abspath(__file__))
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for name, v in out["variants"].items():
        t = v["analytic"]["by_effect"]["+0.030"]
        print(name, "se", round(v["analytic"]["se"], 6), "| P per seed at +0.03: sealed", round(t["sealed"]["per_seed"], 4), "new", round(t["new"]["per_seed"], 4),
              "| both seeds sealed", {k: round(x, 4) for k, x in t["sealed"]["both_seeds_by_rho"].items()}, "new", {k: round(x, 4) for k, x in t["new"]["both_seeds_by_rho"].items()})


if __name__ == "__main__":
    main()
