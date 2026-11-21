"""Amendment 3 power calculation (CPU, numpy).  Reads PRIVGEOM-SAFE's banked arrays READ-ONLY, exactly as `sh5_amend2_power.py` / `sh5_design_calc.py mde` do, and produces NO SH5 outcome (no SH5 number exists).

    python sh5_amend3_power.py --plan work/plan/plan.json --mde work/mde.json --falsefire work/falsefire_sim.json --amend2-power work/amend2_power.json \
        --amend2-lib D:/.../sh5_a3_2026-10-19/amend2_state/sh5_lib.py --out work/amend3_power.json

It answers, BEFORE Amendment 3 is sealed (MM ruling `MM_RULING_SH5.md` s5): C-NULL-LABEL's per-replicate FIRE rule becomes the Amendment-2 CTX-REACHABLE rule (lower CI > 0 AND point >= +0.02) instead of the sealed
lower CI > +0.02, the VOID count rule (any of 3 replicates fires) unchanged.  Two quantities, on the SAME inputs and the SAME stated assumptions as `amend2_power.json` (the banked sign-clean arm's `se = (hi - lo) / (2 x 1.96)`
on the 4,941-set FULL set and the 1,444-set LOCAL ctx subset, re-derived here from the arrays and asserted equal to `work/mde.json`; NO between-seed correlation is involved because each replicate is read on ONE seed):

  1. the FALSE-VOID rate on a clean machinery (true Delta_F = 0 for each noise-trained replicate), sealed fire rule vs the new one;
  2. the VOID POWER against a machinery that manufactures a gain (true Delta_F = +0.02 ... +0.04, headline +0.03), sealed vs new.
P(VOID) over the 3 replicates (`L.NULL_LABEL_REPS`) = P(at least one fires).  The replicates share the logs, so the truth lies between "all three identical" (= the per-replicate probability) and "independent" (1 - (1 - p)^3).

The per-replicate fire THRESHOLD is found by bisecting the REAL function `sh5_lib.null_label_fires` (an interval built as lo = point - 1.96 se, hi = point + 1.96 se) and asserted equal to the closed form max(1.96 se, 0.02): the
arithmetic is the library's rule, not a second copy of it.  The sealed rule's threshold (lower CI > +0.02 -> point > 0.02 + 1.96 se) comes from the Amendment-2-state library file when it is supplied (`--amend2-lib`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_design_calc as DC                                                                           # noqa: E402
import sh5_amend2_power as P2                                                                          # noqa: E402

Z = 1.96
N_REP = L.NULL_LABEL_REPS
EFFECTS = (0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06)


def bisect_threshold(fires, se: float, lo_pt: float = -0.05, hi_pt: float = 0.5) -> float:
    """smallest point estimate at which `fires([ci])` is True, where ci = {point, lo = point - 1.96 se, hi = point + 1.96 se}; `fires` must be monotone in the point (asserted at both ends)."""
    ci = lambda p: {"point": p, "lo": p - Z * se, "hi": p + Z * se}                                    # noqa: E731
    assert not fires([ci(lo_pt)]) and fires([ci(hi_pt)]), "rule is not monotone/bracketed on the search range"
    a, b = lo_pt, hi_pt
    for _ in range(200):
        m = 0.5 * (a + b)
        if fires([ci(m)]):
            b = m
        else:
            a = m
    return b


def load_state(path: str):
    spec = importlib.util.spec_from_file_location("sh5_lib_amend2_state", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def void_bounds(p: float, n: int = N_REP) -> list:
    """[all replicates identical, independent]"""
    return [p, 1.0 - (1.0 - p) ** n]


def rule_table(se: float, th_sealed: float, th_new: float, sd_point: float, effects=EFFECTS) -> dict:
    out = {}
    for eff in effects:
        row = {}
        for name, t in (("sealed", th_sealed), ("new", th_new)):
            p = P2.Q((t - eff) / sd_point)
            row[name] = {"per_replicate_fires": p, "P_VOID_bounds_identical_to_independent": void_bounds(p)}
        out[f"{eff:+.3f}"] = row
    return out


def nested_rows(sim: dict, banked_point: float, effects=(0.0, 0.02, 0.025, 0.03, 0.035, 0.04)) -> dict:
    out = {}
    for e in effects:
        shift = float(e) - banked_point
        pt, lo = sim["point"] + shift, sim["lo"] + shift
        ps, pn = float(((lo > P2.SEALED_LO_MIN)).mean()), float(((lo > L.REACH_LO_MIN) & (pt >= L.REACH_POINT_MIN)).mean())
        out[f"{e:+.3f}"] = {"per_replicate_fires": {"sealed": ps, "new": pn}, "P_VOID_bounds": {"sealed": void_bounds(ps), "new": void_bounds(pn)}}
    return out


def reread_banked_rows(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    nl, pos = d["rows"], d["pos_rows"]
    sealed = lambda ci: bool(ci["lo"] > L.INFO_MARGIN)                                                  # noqa: E731
    return {"noise_trained_replicates": {"n": len(nl), "fire_by_the_new_function_null_label_fires": sum(L.null_label_fires([r["delta_F"]]) for r in nl), "fire_by_the_sealed_rule": sum(sealed(r["delta_F"]) for r in nl),
                                         "n_lower_ci_gt_0": sum(r["delta_F"]["lo"] > 0 for r in nl), "max_point": max(r["delta_F"]["point"] for r in nl), "min_point": min(r["delta_F"]["point"] for r in nl)},
            "positive_control_replicates_branch_reads_the_ctx": {"n": len(pos), "fire_by_the_new_function_null_label_fires": sum(L.null_label_fires([r["delta_F"]]) for r in pos), "fire_by_the_sealed_rule": sum(sealed(r["delta_F"]) for r in pos),
                                                                "points": [r["delta_F"]["point"] for r in pos]},
            "note": "CPU synthetic world (sh5_sim_falsefire.py, banked work/falsefire_sim.json); 16 noise-trained replicates + 4 branch-reads-the-ctx replicates.  Valid as a bound on the machinery's false-fire rate in that world, not for the real rig's training variance (unmeasured, H-ESTIM-SEED-1)."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mde", required=True)
    ap.add_argument("--falsefire", required=True)
    ap.add_argument("--amend2-power", required=True)
    ap.add_argument("--amend2-lib", default=None)
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
    a2 = json.load(open(a.amend2_power, encoding="utf-8"))
    old = load_state(a.amend2_lib) if a.amend2_lib else None
    out = {"purpose": "Amendment 3 power: C-NULL-LABEL fire rule, sealed (lower CI > +0.02, the rule Amendment 2 kept) vs NEW (= the Amendment-2 CTX-REACHABLE rule: lower CI > 0 AND point >= +0.02): the false-VOID rate on a clean machinery and the VOID power against a machinery that manufactures a gain",
           "evidence_class": "MEASURED inputs (PRIVGEOM banked arrays + banked CPU false-fire rows), ESTIMATED probabilities (normal approximation + nested bootstrap)",
           "rule_constants": {"sealed_fire": {"lower_ci_gt": L.INFO_MARGIN}, "new_fire": {"lower_ci_gt": L.REACH_LO_MIN, "point_ge": L.REACH_POINT_MIN}, "n_replicates": N_REP, "replicate_seed_layout": "rep r is read on seed r % 2 (reps 0, 2 on seed 0; rep 1 on seed 1)", "true_effect_sized_for": L.POWER_EFFECT},
           "assumptions": ("the SAME as amend2_power.json: each replicate's POINT estimate has marginal sd = se (the banked sign-clean arm's interval half-width / 1.96: CONSERVATIVE for a true zero, where the interval is far narrower, PRIVGEOM RESULT 5.3 ~0.001 wide and the 16 noise-trained CPU replicates scattered with sd 0.0006); "
                           "the three replicates share the logs, so P(VOID) lies between 'all identical' (= per replicate) and 'independent' (1 - (1 - p)^3); sensitivity: training variance EQUAL to the sampling variance on top (point sd sqrt(2) se, interval still se wide); a nested-bootstrap cross-check that does not assume a normal, symmetric interval. "
                           "No between-seed correlation is involved (one seed per replicate)."),
           "variants": {}}
    for name, key, sel in (("FULL", "full_4941_sets", np.ones(len(zL), bool)), ("LOCAL", "local_ctx_sets", local)):
        li = A["log_idx"][sel]
        u, loc = np.unique(li, return_inverse=True)
        an = L.Analyzer(lab_dac[sel], A["code"][sel], loc, len(u))
        stD, stL = an.stats(zD[sel]), an.stats(zL[sel])
        d = an.diff(stD, stL, "Q4")
        se = (d["hi"] - d["lo"]) / (2 * Z)
        assert abs(se - sealed_mde[key]["se_from_interval"]) < 1e-12, f"{name}: se does not reproduce the sealed MDE input"
        assert abs(se - a2["variants"][name]["analytic"]["se"]) < 1e-12, f"{name}: se does not reproduce Amendment 2's"
        th_new = bisect_threshold(L.null_label_fires, se)
        assert abs(th_new - max(Z * se, L.REACH_POINT_MIN)) < 1e-9, (th_new, max(Z * se, L.REACH_POINT_MIN))
        th_sealed_closed = L.INFO_MARGIN + Z * se
        sealed_fires = (old.null_label_fires if old is not None else (lambda cs: bool(len(cs) > 0 and all(c["lo"] > L.INFO_MARGIN for c in cs))))
        th_sealed = bisect_threshold(sealed_fires, se)
        assert abs(th_sealed - th_sealed_closed) < 1e-9, (th_sealed, th_sealed_closed)
        sd2 = math.sqrt(2.0) * se
        sim = P2.nested(*stD["Q4"], *stL["Q4"], a.R, a.B, 20261019)
        fft = P2.false_fire_table(se)
        v = {"n_sets": int(sel.sum()), "n_logs": int(len(u)), "banked_sign_clean_arm_delta_F": d, "se": se,
             "thresholds_point_needed_to_fire": {"sealed_bisected_on_the_AMEND2_STATE_null_label_fires": th_sealed if old is not None else None, "sealed_closed_form_0.02_plus_1.96se": th_sealed_closed, "sealed_bisected": th_sealed,
                                                 "new_bisected_on_the_REAL_null_label_fires": th_new, "new_closed_form_max(1.96se,0.02)": max(Z * se, L.REACH_POINT_MIN),
                                                 "binding_conjunct_of_new_rule": "lower CI > 0" if Z * se > L.REACH_POINT_MIN else "point >= +0.02"},
             "analytic_sampling_only_se": rule_table(se, th_sealed, th_new, se),
             "analytic_training_var_equals_sampling_var": {"point_sd": sd2, **{"by_effect": rule_table(se, th_sealed, th_new, sd2)}},
             "nested_bootstrap": {"R_outer": a.R, "B_inner": a.B, "by_effect": nested_rows(sim, d["point"])},
             "false_VOID_clean_machinery_bracketing_rows_point_sd_below_se": {k: {"per_replicate_new": r["new"]["per_seed"], "per_replicate_sealed": r["sealed"]["per_seed"]} for k, r in fft.items()}}
        # cross-checks against the banked Amendment-2 numbers (same inputs => the per-replicate probability IS Amendment 2's per-seed CTX-REACHABLE power under the new rule)
        a2_t = a2["variants"][name]["analytic"]["by_effect"]
        for eff in EFFECTS:
            e = f"{eff:+.3f}"
            assert abs(v["analytic_sampling_only_se"][e]["new"]["per_replicate_fires"] - a2_t[e]["new"]["per_seed"]) < 1e-9, (name, e)
            assert abs(v["analytic_sampling_only_se"][e]["sealed"]["per_replicate_fires"] - a2_t[e]["sealed"]["per_seed"]) < 1e-9, (name, e)
        v["reproduces_amend2_per_seed_probabilities"] = True
        out["variants"][name] = v
    out["banked_cpu_false_fire_rows_read_by_the_new_function"] = reread_banked_rows(a.falsefire)
    head = {}
    for name, v in out["variants"].items():
        t0, t3 = v["analytic_sampling_only_se"]["+0.000"], v["analytic_sampling_only_se"]["+0.030"]
        s0, s3 = v["analytic_training_var_equals_sampling_var"]["by_effect"]["+0.000"], v["analytic_training_var_equals_sampling_var"]["by_effect"]["+0.030"]
        n0, n3 = v["nested_bootstrap"]["by_effect"]["+0.000"], v["nested_bootstrap"]["by_effect"]["+0.030"]
        head[name] = {"se": v["se"], "n_logs": v["n_logs"], "binding_conjunct": v["thresholds_point_needed_to_fire"]["binding_conjunct_of_new_rule"], "point_needed_to_fire": {"sealed": v["thresholds_point_needed_to_fire"]["sealed_bisected"], "new": v["thresholds_point_needed_to_fire"]["new_bisected_on_the_REAL_null_label_fires"]},
                      "FALSE_VOID_clean_machinery(true_delta_F=0)": {"sampling_only_conservative_se": {"per_replicate": {"sealed": t0["sealed"]["per_replicate_fires"], "new": t0["new"]["per_replicate_fires"]},
                                                                                                        "P_VOID_bounds": {"sealed": t0["sealed"]["P_VOID_bounds_identical_to_independent"], "new": t0["new"]["P_VOID_bounds_identical_to_independent"]}},
                                                                    "training_var_equals_sampling_var": {"per_replicate": {"sealed": s0["sealed"]["per_replicate_fires"], "new": s0["new"]["per_replicate_fires"]},
                                                                                                         "P_VOID_bounds": {"sealed": s0["sealed"]["P_VOID_bounds_identical_to_independent"], "new": s0["new"]["P_VOID_bounds_identical_to_independent"]}},
                                                                    "nested_bootstrap": n0["P_VOID_bounds"]},
                      "VOID_POWER_machinery_manufactures_+0.03": {"sampling_only": {"per_replicate": {"sealed": t3["sealed"]["per_replicate_fires"], "new": t3["new"]["per_replicate_fires"]},
                                                                                    "P_VOID_bounds": {"sealed": t3["sealed"]["P_VOID_bounds_identical_to_independent"], "new": t3["new"]["P_VOID_bounds_identical_to_independent"]}},
                                                                  "training_var_equals_sampling_var": {"per_replicate": {"sealed": s3["sealed"]["per_replicate_fires"], "new": s3["new"]["per_replicate_fires"]},
                                                                                                       "P_VOID_bounds": {"sealed": s3["sealed"]["P_VOID_bounds_identical_to_independent"], "new": s3["new"]["P_VOID_bounds_identical_to_independent"]}},
                                                                  "nested_bootstrap": n3["P_VOID_bounds"]}}
    out["headline"] = head
    out["amend2_state_library_sha256"] = L.sha256_file(a.amend2_lib) if a.amend2_lib else None
    out["power_script_sha256"] = L.sha256_file(os.path.abspath(__file__))
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for name, h in head.items():
        f, p = h["FALSE_VOID_clean_machinery(true_delta_F=0)"], h["VOID_POWER_machinery_manufactures_+0.03"]
        print(name, "se", round(h["se"], 6), "| false VOID (conservative se) sealed", [f"{x:.2e}" for x in f["sampling_only_conservative_se"]["P_VOID_bounds"]["sealed"]], "new", [round(x, 4) for x in f["sampling_only_conservative_se"]["P_VOID_bounds"]["new"]],
              "| VOID power at +0.03 sealed", [round(x, 3) for x in p["sampling_only"]["P_VOID_bounds"]["sealed"]], "new", [round(x, 3) for x in p["sampling_only"]["P_VOID_bounds"]["new"]])


if __name__ == "__main__":
    main()
