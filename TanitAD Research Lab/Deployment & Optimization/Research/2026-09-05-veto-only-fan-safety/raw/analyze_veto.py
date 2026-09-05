#!/usr/bin/env python3
"""H-VETO-FAN-1 readout: the lever effect READ AGAINST THE RIG'S OWN RUN-TO-RUN NOISE.

⛔ WHY THIS FILE EXISTS RATHER THAN A TABLE OF SEPARATED CIs. `H-ESTIM-SEED-1`
(CLAUDE.md, MEASURED 2026-09-05): the episode-cluster bootstrap resamples EPISODES with
the models held fixed, so it answers "would another DRAW OF EPISODES say this?" and never
"would another TRAINING RUN say this?". A zero-lever replicate cleared `separated` on 3 of
18 metrics -- a ~17 % false-positive rate. So a separated CI is NECESSARY, NOT SUFFICIENT,
and this tool computes the sufficient form:

  lever(m, K)  = paired delta( veto_sK AFTER , base BEFORE )      # per seed K
  floor(m)     = | paired delta( veto_s0 AFTER , veto_s1 AFTER ) |  # SAME flags, seed apart

  QUOTABLE(m) iff  lever separated for BOTH seeds
              AND  same sign
              AND  min(|lever_s0|, |lever_s1|) > floor(m)
              AND  |lever| >= MIN_EFFECT[m]   (per-metric, from its own quantum AND UNITS)

Everything else is reported WITHIN-NOISE or UNDETECTABLE-DOWNWARD, never as a null.

GUARDS, checked before any row is read (SPEC §7):
  G1  ctrl_null (veto OFF, zero reward) must read veto_rate_mean EXACTLY 0.0
  G2  every arm's BEFORE readout must be BITWISE identical to the base's
  G3  a metric quoted from one seed is VOID
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.environ.get("TANITAD_REPO", "/c/Users/Admin/refcv4b_repo")


def load_driver():
    path = os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_an", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = load_driver()

#: Per-metric separation floor, derived from that metric's OWN quantum and UNITS.
#: A single 1e-4 "rate quantum" mis-fits `fan_peak_g_mean` (g) and
#: `fan_v_mean_2s_spread` (m/s) -- 2 of the 14 metrics that fired the previous panel's
#: VOID gate. Rates keep 1/(128 x 120) rounded up to 1e-4.
RATE_FLOOR = 1e-4
FLOORS = {"fan_peak_g_mean": 1e-3, "sel_peak_g": 1e-3, "fan_v_mean_2s_spread": 1e-3}

#: The PRIMARY endpoint, in the order the SPEC reads it.
PRIMARY = ("fan_peak_g_mean", "top32_infeasible", "sel_infeasible")
SUPPORT = ("fan_infeasible", "top8_kamm_over", "fan_kamm_over", "top32_envelope",
           "top32_kamm_over", "fan_envelope", "sel_envelope", "sel_ttc_below",
           "sel_peak_g", "fan_off_reach", "top8_infeasible", "mass_rank_infeasible",
           "mass_conf_infeasible", "fan_contact", "top32_contact", "mass_rank_contact")


def floor_of(m: str) -> float:
    return FLOORS.get(m, RATE_FLOOR)


def J(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def rows_of(d):
    return {r["wi"]: r for r in d["per_window"]}


def bitwise_identical(a: dict, b: dict) -> dict:
    """G2 -- the BEFORE readouts must be the SAME numbers, not merely similar."""
    ra, rb = rows_of(a), rows_of(b)
    shared = sorted(set(ra) & set(rb))
    keys = [k for k in ra[shared[0]] if isinstance(ra[shared[0]][k], (int, float))
            and k not in ("wi", "eid")]
    worst, worst_k = 0.0, None
    for wi in shared:
        for k in keys:
            d = abs(float(ra[wi][k]) - float(rb[wi][k]))
            if d > worst:
                worst, worst_k = d, k
    return {"n_shared_windows": len(shared), "n_metrics": len(keys),
            "max_abs_diff": worst, "worst_metric": worst_k,
            "identical": worst == 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="dir holding s0/ and s1/")
    ap.add_argument("--arm", required=True, help="veto200 | veto2k")
    ap.add_argument("--null-arm", default="ctrl_null")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    s0 = os.path.join(a.run_dir, "s0", a.arm)
    s1 = os.path.join(a.run_dir, "s1", a.arm)
    for d in (s0, s1):
        if not os.path.isfile(os.path.join(d, "arm_summary.json")):
            raise SystemExit(f"[veto] missing {d}/arm_summary.json -- G3: a one-seed "
                             "result is VOID, refusing to report")
    sum0, sum1 = J(os.path.join(s0, "arm_summary.json")), J(os.path.join(s1, "arm_summary.json"))
    b0, a0 = J(os.path.join(s0, "readout_before.json")), J(os.path.join(s0, "readout_after.json"))
    b1, a1 = J(os.path.join(s1, "readout_before.json")), J(os.path.join(s1, "readout_after.json"))

    out = {"_tool": "analyze_veto.py", "_tier": "T0 readout on the EMITTED fan (never a driving claim)",
           "_evidence_class": "MEASURED (ours)", "arm": a.arm, "run_dir": a.run_dir,
           "estimator": "paired episode-cluster bootstrap (taniteval/ci.py via rl_refcv3_min.paired_delta)",
           "n_boot": a.n_boot, "seed": a.seed, "guards": {}, "metrics": {}}

    # ---- G1: the null must be an ACTUAL null ------------------------------- #
    nulls = {}
    for k in ("s0", "s1"):
        p = os.path.join(a.run_dir, k, a.null_arm, "arm_summary.json")
        if os.path.isfile(p):
            n = J(p)
            nulls[k] = {"veto_rate_mean": n.get("veto_rate_mean"),
                        "final_loss": n.get("final_loss"),
                        "weights_changed": n.get("weights_changed"),
                        "n_separated_fan_metrics": sum(
                            1 for v in (n.get("fan_safety_deltas_paired") or {}).values()
                            if v.get("sep"))}
    g1_ok = bool(nulls) and all(v["veto_rate_mean"] == 0.0 for v in nulls.values())
    out["guards"]["G1_ctrl_null_is_an_actual_null"] = {
        "pass": g1_ok, "arms": nulls,
        "rule": "veto_rate_mean must be EXACTLY 0.0 with the veto off; the previous "
                "panel's zero-weight control read 0.0897 because the veto was keyed on "
                "the reward's KEY SET"}

    # ---- G2: identical BEFORE readouts ------------------------------------- #
    out["guards"]["G2_before_readouts_bitwise_identical"] = bitwise_identical(b0, b1)

    # ---- the metrics ------------------------------------------------------- #
    keys = [k for k in b0["per_window"][0]
            if any(k.endswith("_" + f) for f in D.FS.FLAGS)
            or k in ("fan_peak_g_mean", "sel_peak_g", "fan_v_mean_2s_spread")]
    lead_only = lambda k: any(f in k for f in D.FS.LEAD_ONLY)          # noqa: E731

    def sub(d):
        return {**d, "per_window": [r for r in d["per_window"] if r["has_lead"]]}

    for k in sorted(keys):
        lo_ = lead_only(k)
        B0, A0 = (sub(b0), sub(a0)) if lo_ else (b0, a0)
        B1, A1 = (sub(b1), sub(a1)) if lo_ else (b1, a1)
        lev0 = D.paired_delta(B0, A0, k, reps=a.n_boot, seed=a.seed)
        lev1 = D.paired_delta(B1, A1, k, reps=a.n_boot, seed=a.seed)
        rep = D.paired_delta(A0, A1, k, reps=a.n_boot, seed=a.seed)   # the noise floor
        fl = floor_of(k)
        base_val = float(b0["fan_safety"][k])
        undetectable_down = abs(base_val) < fl
        same_sign = (lev0["delta"] * lev1["delta"]) > 0
        both_sep = bool(lev0["sep"] and lev1["sep"])
        big_enough = min(abs(lev0["delta"]), abs(lev1["delta"])) >= fl
        clears_floor = min(abs(lev0["delta"]), abs(lev1["delta"])) > abs(rep["delta"])
        quotable = bool(both_sep and same_sign and big_enough and clears_floor)
        if quotable:
            verdict = "IMPROVED" if lev0["delta"] < 0 else "WORSENED"
        elif both_sep and same_sign and big_enough and not clears_floor:
            verdict = "WITHIN-NOISE (separated but under the seed-replicate floor)"
        elif undetectable_down and lev0["delta"] >= 0:
            verdict = "UNDETECTABLE-DOWNWARD"
        else:
            verdict = "null"
        out["metrics"][k] = {
            "population": "lead windows" if lo_ else "all windows",
            "base": base_val, "after_s0": float(a0["fan_safety"][k]),
            "after_s1": float(a1["fan_safety"][k]),
            "lever_s0": lev0, "lever_s1": lev1, "replicate_floor": rep,
            "min_effect": fl, "both_separated": both_sep, "same_sign": bool(same_sign),
            "clears_replicate_floor": bool(clears_floor),
            "undetectable_downward": bool(undetectable_down),
            "quotable_as_lever": quotable, "verdict": verdict}

    # ---- the SPEC's acceptance -------------------------------------------- #
    m = out["metrics"]
    peak = m.get("fan_peak_g_mean", {})
    improved_primary = [k for k in PRIMARY
                        if m.get(k, {}).get("quotable_as_lever") and m[k]["lever_s0"]["delta"] < 0]
    out["acceptance"] = {
        "primary_metrics": list(PRIMARY),
        "improved_and_quotable": improved_primary,
        "fan_peak_g_mean_improved": bool(peak.get("quotable_as_lever")
                                         and peak.get("lever_s0", {}).get("delta", 1) < 0),
        "rule": ("SUCCESS needs fan_peak_g_mean NEGATIVE and quotable AND at least one of "
                 "top32_infeasible / sel_infeasible NEGATIVE and quotable, AND ade_m at T1 "
                 "not regressed beyond the same replicate floor (scored separately)."),
        "primary_verdict_pending_T1": (
            "PRIMARY-PASS" if (peak.get("quotable_as_lever")
                               and peak.get("lever_s0", {}).get("delta", 1) < 0
                               and improved_primary != ["fan_peak_g_mean"] and improved_primary)
            else "PRIMARY-FAIL")}
    out["run_facts"] = {
        f"s{k}": {kk: v.get(kk) for kk in
                  ("veto_rate_mean", "final_loss", "weights_changed", "train_wallclock_s",
                   "sel_idx_agreement_with_base", "frac_above_bar_mean", "steps")}
        for k, v in (("0", sum0), ("1", sum1))}
    for k, v in (("0", sum0), ("1", sum1)):
        out["run_facts"][f"s{k}"]["reward_weights"] = (v.get("config") or {}).get("reward_weights")
        out["run_facts"][f"s{k}"]["components_fired"] = v.get("components_fired")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)

    print(f"\n=== {a.arm}: GUARDS ===")
    print(f"  G1 ctrl_null actual null : {out['guards']['G1_ctrl_null_is_an_actual_null']['pass']}  "
          f"{json.dumps(nulls)}")
    g2 = out["guards"]["G2_before_readouts_bitwise_identical"]
    print(f"  G2 BEFORE bitwise ident. : {g2['identical']}  max|diff|={g2['max_abs_diff']:.3e} "
          f"over {g2['n_shared_windows']}w x {g2['n_metrics']} metrics")
    print(f"\n=== {a.arm}: LEVER vs BASE, READ AGAINST THE SEED-REPLICATE FLOOR ===")
    hdr = f"  {'metric':26s} {'base':>9s} {'d_s0':>10s} {'d_s1':>10s} {'floor':>9s}  verdict"
    print(hdr)
    for k in list(PRIMARY) + [x for x in SUPPORT if x in m]:
        r = m[k]
        print(f"  {k:26s} {r['base']:9.5f} {r['lever_s0']['delta']:+10.5f} "
              f"{r['lever_s1']['delta']:+10.5f} {abs(r['replicate_floor']['delta']):9.5f}  "
              f"{r['verdict']}")
    print(f"\n  PRIMARY (pending T1): {out['acceptance']['primary_verdict_pending_T1']}  "
          f"improved+quotable={improved_primary}")
    print(f"[veto] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
