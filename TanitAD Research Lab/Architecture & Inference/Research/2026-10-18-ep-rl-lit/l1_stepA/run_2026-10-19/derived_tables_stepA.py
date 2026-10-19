#!/usr/bin/env python3
"""L1 Step A: DERIVED tables from the banked arrays (CPU only, numpy only, seconds).  POST-HOC and DESCRIPTIVE: nothing here is a pre-registered quantity and nothing here
changes the outcome.  It does not re-run steps 3-7; it re-reads the four banked arrays and the bundle.

Three jobs, in this order (the controls come first so a wrong reader cannot produce a table):
  C1  REPRODUCE the headline: stepA.analyse() on the banked arrays must equal RESULT.json (all / lon / lat A1..A5 + tactical) to 1e-12.
  C2  INDEPENDENT G2b: the bundle's fan (repaired), logits and pick equal the banked P2 table bit for bit on the 400 tokens (np.array_equal), without the GPU.
  T   the descriptive table: A1 (EP concordance) by speed-scale value and by anchor type, with the SAME paired log-cluster bootstrap (24 logs, 10,000, seed 20260927,
      the same draws as the headline), ratio of sums over whole logs.  A cell whose resample has an empty denominator is skipped and counted (`n_valid_resamples`).

usage: python -I derived_tables_stepA.py --run-dir D:/Projects/TanitAD-artifacts/l1_stepA_run --code-dir C:/Users/Admin/par18/l1run/code --out derived_tables.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="D:/Projects/TanitAD-artifacts/l1_stepA_run")
    ap.add_argument("--code-dir", default="C:/Users/Admin/par18/l1run/code")
    ap.add_argument("--table", default="D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz")
    ap.add_argument("--proposals20", default="D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/proposals.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import stepA as S

    W = a.run_dir
    b, pe, sc, lab = (S.load_npz(os.path.join(W, f)) for f in ("bundle/bundle.npz", "perts.npz", "scores.npz", "labels.npz"))
    ref = json.load(open(os.path.join(W, "RESULT.json"), encoding="utf-8"))
    names = sorted(set(str(x) for x in b["log"]))
    P, li, kinds = S.assemble_pairs(b, pe, sc, lab, names)
    n_logs = len(names)
    N, Q = pe["anchor"].shape
    out: dict = {"post_hoc": True, "descriptive": True, "n_logs": n_logs, "bootstrap": {"B": S.N_BOOT, "seed": S.SEED, "unit": "log", "same_draws_as_headline": True,
                 "answers": "another draw of LOGS only; one training seed (H-ESTIM-SEED-1); no search was run"}}

    # ---- C1 reproduce the headline
    res = S.analyse(P, li, n_logs, S.N_BOOT, S.SEED, "interval_aware")
    worst = 0.0
    for tag in ("all", "lon", "lat"):
        for q, d in res[tag].items():
            for f in ("point", "lo", "hi", "den", "n_valid_resamples"):
                worst = max(worst, abs(float(d[f]) - float(ref[tag][q][f])))
    worst = max(worst, abs(res["tactical"]["share_changing_command_class"] - ref["tactical"]["share_changing_command_class"]))
    out["C1_reproduce_headline"] = {"max_abs_diff_vs_RESULT_json": worst, "passed": bool(worst <= 1e-12), "outcome": res["outcome"], "outcome_if_literal": res["outcome_if_literal"],
                                    "outcome_in_RESULT_json": ref["outcome"], "outcome_literal_in_RESULT_json": ref["outcome_if_literal"]}
    if not out["C1_reproduce_headline"]["passed"] or res["outcome"] != ref["outcome"]:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ZZDERIVED_FAIL C1 headline not reproduced", worst)
        return 1

    # ---- C2 independent G2b (no GPU): the bundle forward == the banked P2 table, bit for bit
    t = np.load(a.table, allow_pickle=True)
    ix = {str(k): i for i, k in enumerate(t["token"])}
    rows = [ix[str(x)] for x in b["token"]]
    p20 = np.load(a.proposals20, allow_pickle=True)
    P20 = p20["proposals20"]
    pi = {str(k): i for i, k in enumerate(p20["token"])}
    rep = np.asarray(b["traj_raw"], np.float32).copy()
    rep[:, :, -1, 2] = rep[:, :, -2, 2]                                         # planner.repair_last_heading (A7)
    fan_ok = np.array([np.array_equal(rep[n], P20[pi[str(b["token"][n])]]) for n in range(len(rows))])
    logit_ok = np.array([np.array_equal(np.asarray(b["logits"][n], np.float32), np.asarray(t["logits"][rows[n]], np.float32)) for n in range(len(rows))])
    pick_ok = np.asarray(b["pick"]) == np.asarray(t["pick"][rows])
    out["C2_g2b_independent"] = {"n_tokens": int(len(rows)), "fan_repaired_equal_banked": int(fan_ok.sum()), "logits_equal_banked": int(logit_ok.sum()), "pick_equal_banked": int(pick_ok.sum()),
                                 "passed": bool(fan_ok.all() and logit_ok.all() and pick_ok.all())}

    # ---- T the descriptive table (flat order of assemble_pairs == (token, job) row-major; no pair was dropped, asserted)
    assert len(kinds) == N * Q, "pairs were dropped; the flat-order mapping below would be wrong"
    flat = lambda x: np.asarray(x).reshape(-1)
    param, is_top = flat(pe["param"]), flat(pe["is_top"]).astype(bool)
    assert np.array_equal(flat(pe["kind"]), kinds)
    terms = S.admission_terms(P)
    idx = S.boot_indices(n_logs, S.N_BOOT, S.SEED)
    lon, lat = kinds == S.KIND_LON, kinds == S.KIND_LAT

    ep_a, ep_p = np.asarray(P["a_sub"], np.float64)[:, 2], np.asarray(P["p_sub"], np.float64)[:, 2]
    lg_a, lg_p = np.asarray(P["a_logit"], np.float64)[:, 2], np.asarray(P["p_logit"], np.float64)[:, 2]
    qual0 = np.abs(ep_p - ep_a) >= S.THR_TRUE_PAIR - 1e-12                       # the A1 pair rule

    def cell(m, key="A1"):
        s = S.per_log_sums({key: terms[key]}, li, n_logs, m)[key]
        d = S.ratio_ci(s[0], s[1], idx)
        d["n_pairs"] = int(np.asarray(m).sum())
        d["n_logs_with_den"] = int((s[1] > 0).sum())
        q = np.asarray(m, bool) & qual0                                          # raw direction shares among the A1-qualifying pairs of this cell (no estimator, descriptive)
        d["share_true_EP_up"] = float(np.mean(ep_p[q] > ep_a[q])) if q.any() else float("nan")
        d["share_scorer_EP_up"] = float(np.mean(lg_p[q] > lg_a[q])) if q.any() else float("nan")
        return d

    rows_out = []

    def add(group, label, m, key="A1"):
        r = cell(m, key)
        r.update({"group": group, "label": label})
        rows_out.append(r)

    for s in S.SPEED_SCALES:
        add("speed_scale (lon)", f"s={s:g}", lon & np.isclose(param, s))
    add("speed_scale (lon)", "s<1 (slower)", lon & (param < 1))
    add("speed_scale (lon)", "s>1 (faster)", lon & (param > 1))
    for d in S.LAT_OFFSETS:
        add("lateral offset (lat)", f"d={d:+g} m", lat & np.isclose(param, d))
    add("anchor type (lon+lat)", "scorer top-8", is_top)
    add("anchor type (lon+lat)", "random-8", ~is_top)
    add("anchor type (lon only)", "scorer top-8", is_top & lon)
    add("anchor type (lon only)", "random-8", ~is_top & lon)
    add("anchor type (lat only)", "scorer top-8", is_top & lat)
    add("anchor type (lat only)", "random-8", ~is_top & lat)
    for s in S.SPEED_SCALES:
        add("speed_scale x anchor (lon)", f"s={s:g} / top-8", lon & np.isclose(param, s) & is_top)
        add("speed_scale x anchor (lon)", f"s={s:g} / random-8", lon & np.isclose(param, s) & ~is_top)
    add("all pairs", "A1 headline (check)", np.ones_like(lon))
    out["T_A1_descriptive"] = rows_out
    # tactical share (the pre-registered tactical quantity): RESULT.json carries a point only; the same log-cluster draws give an interval here (derived, NOT in RESULT.json)
    tact = np.asarray(P["tact"], np.float64)
    out["T_tactical_share_changing_command_class"] = {}
    for tag, m in (("all", np.ones_like(lon)), ("lon", lon), ("lat", lat)):
        nb = np.bincount(li[m], weights=tact[m], minlength=n_logs)
        db = np.bincount(li[m], weights=np.ones(int(m.sum())), minlength=n_logs)
        r = S.ratio_ci(nb, db, idx)
        r["n_changing"] = int(tact[m].sum())
        r["n_pairs"] = int(m.sum())
        out["T_tactical_share_changing_command_class"][tag] = r
    # the same strata for A2 / A4 are NOT produced: the brief asks for A1 only, and every extra stratum is another post-hoc look.

    # ---- I inventory of the labelled off-proposal pairs (what prereg section 4's O2 row hands to L2): RAW COUNTS, no estimator
    al, pl = np.asarray(P["a_logit"], np.float64), np.asarray(P["p_logit"], np.float64)
    ae, pe_ = np.asarray(P["a_sub"], np.float64)[:, 2], np.asarray(P["p_sub"], np.float64)[:, 2]
    qual = np.abs(pe_ - ae) >= S.THR_TRUE_PAIR - 1e-12
    ds, dt = np.sign(pl[:, 2] - al[:, 2]), np.sign(pe_ - ae)
    inv = {"n_pairs": int(len(kinds)), "n_pairs_lon": int(lon.sum()), "n_pairs_lat": int(lat.sum()),
           "A1_qualifying_pairs_(|dEP|>=0.05)": int(qual.sum()), "A1_qualifying_lon": int((qual & lon).sum()), "A1_qualifying_lat": int((qual & lat).sum()),
           "A1_concordant_strict": int((qual & (ds == dt) & (ds != 0)).sum()), "A1_discordant_strict": int((qual & (ds != dt) & (ds != 0)).sum()), "A1_scorer_ties": int((qual & (ds == 0)).sum()),
           "A4_scorer_endorsed_pairs_(aggregate_rise>=0.02)": int(round(res["all"]["A4"]["den"])), "A4_endorsed_and_true_PDMS_fall>=0.10": int(round(res["all"]["A4"]["point"] * res["all"]["A4"]["den"])),
           "A5_endorsed_and_true_NC_or_DAC_1to0": int(round(res["all"]["A5"]["point"] * res["all"]["A5"]["den"])), "true_NC_or_DAC_1to0_breaches_(A5b_den)": int(round(res["all"]["A5b"]["den"])),
           "note": "labelled pairs available as L2 training data (prereg section 4, O2 row); counts only, no estimator; the label is the true PDM sub-score of the perturbed plan"}
    out["I_pair_inventory"] = inv

    # ---- N numerics bound (C9): the CPU fp32 / bf16-ctx scoring differs from the GPU fp32 forward by at most `eps` logits (G2b_cross_device).  A pair's EP ordering can only
    #          flip if |dEP logit| < 2*eps, so the share of A1-qualifying pairs inside that band BOUNDS how far A1 could move had the scoring been bit-exact.  Counts only.
    eps = float(json.load(open(os.path.join(W, "scores.npz.gates.json"), encoding="utf-8"))["G2b_cross_device"][0]["max_abs_diff_vs_bundle_logits"])
    dl = np.abs(pl[:, 2] - al[:, 2])
    close = qual & (dl < 2 * eps)
    out["N_numerics_bound"] = {"eps_max_abs_logit_diff_cpu_vs_gpu": eps, "band": 2 * eps, "A1_qualifying_pairs": int(qual.sum()), "A1_pairs_inside_band": int(close.sum()),
                               "max_possible_A1_shift_(pairs_in_band/qualifying)": float(close.sum() / qual.sum()),
                               "A1_gate": S.GATE_A1, "A1_upper_95_headline": float(ref["all"]["A1"]["hi"]),
                               "note": "an upper bound on how far a bit-exact GPU re-score could move the A1 POINT estimate; it does not bound the interval and is a bound, not a re-score"}

    # ---- G gate numbers, read from the banked gate JSONs (so the RESULT.md quotes code output, not a hand count)
    gj = json.load(open(os.path.join(W, "scores.npz.gates.json"), encoding="utf-8"))
    g1 = json.load(open(os.path.join(W, "labels.npz.g1.json"), encoding="utf-8"))
    c2 = np.array([x["max_abs_change_on_other_63"] for x in gj["G2c"]])
    out["G_summary"] = {"G2a": {"n": len(gj["G2a"]), "passed": int(sum(bool(x["passed"]) for x in gj["G2a"])), "max_abs_diff": float(max(x["max_abs_diff"] for x in gj["G2a"])),
                                "input_identical": int(sum(bool(x["input_identical"]) for x in gj["G2a"]))},
                        "G2c": {"n": len(gj["G2c"]), "premise_true": int(sum(bool(x["premise_true"]) for x in gj["G2c"])), "min_change": float(c2.min()), "median_change": float(np.median(c2)), "max_change": float(c2.max())},
                        "G3": {"n": len(gj["G3"]), "passed": int(sum(bool(x["passed"]) for x in gj["G3"])), "slots_exactly_expected": int(sum(x["slots_changed"] == [x["expected_slot"]] for x in gj["G3"]))},
                        "G2b_cross_device": gj["G2b_cross_device"], "G1": g1}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZDERIVED_OK", "C1", out["C1_reproduce_headline"]["passed"], "C2", out["C2_g2b_independent"]["passed"], len(rows_out), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
