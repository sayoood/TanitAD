"""build_verdict.py -- applies PREREG_V8F_N1DW.md (sha256 59f2b7f9) section 7 with AMENDMENT_N1DW_1 (bff20b4d) to D1-HO + eval-141, mechanically, and writes RESULT_N1DW_VERDICT.json.
Reads only existing result files (zero training, zero dumps). Every number it writes is copied from a named input file; the rules are the card's rows as written.
Usage: python build_verdict.py"""
import hashlib
import json
import os

V = "F:/TanitAD-scratch/n1dw/read/verdict"
IN = {
    "d1ho": "F:/TanitAD-scratch/n1dw/read/RESULT_N1DW_D1HO.json",
    "e141": "F:/TanitAD-scratch/n1dw/read/eval141/out/RESULT_N1DW_EVAL141.json",
    "contrast": "F:/TanitAD-scratch/n1dw/read/eval141/out/N1DW_STRATEGIC_contrast.json",
    "s1_w2": V + "/N1DW2_STRATEGIC_S1slot.json",
    "s1_w4": V + "/N1DW4_STRATEGIC_S1slot.json",
    "s1_n1r": V + "/N1R_STRATEGIC_S1slot.json",
    "ctx_nofog_w4": "F:/TanitAD-scratch/n1dw/read/eval141/out/N1DW4_CTX_nofog.json",
    "navrbd": "F:/TanitAD-scratch/strat1/navrbd/read/out/RESULT_NAVRBD.json",
}
W, S = (2, 4), (0, 1)
ARCS = (30, 50, 80)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def r4(x):
    return None if x is None else round(float(x), 4)


def ci(r):
    if r.get("status") != "OK":
        return {"status": r.get("status"), "reason": r.get("reason"), "n_windows": r.get("n_windows"), "n_episodes": r.get("n_episodes")}
    o = {"delta": r4(r["delta"]), "ci95": [r4(r["lo"]), r4(r["hi"])], "separated": bool(r["separated"]), "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]}
    for k in ("gain", "cost", "floor", "bar"):
        if k in r and r[k] is not None:
            o[k] = r4(r[k]) if k in ("floor", "bar") else bool(r[k])
    return o


def main():
    J = {k: json.load(open(p, encoding="utf-8")) for k, p in IN.items()}
    d1, e1, ct = J["d1ho"], J["e141"], J["contrast"]
    bar = {w: d1["floor"]["bar_by_weight"][f"w{w}"] for w in W}
    dec = {w: {s: d1["D1_HO_deciding_N1DW-N1R"][f"N1DW{w}_s{s}"] for s in S} for w in W}
    bes = {w: {s: d1["D1_HO_beside_N1DW-N1D"][f"N1DW{w}_s{s}"] for s in S} for w in W}
    # GAIN at a seed and weight (5.3): delta >= bar and lo > 0 -- recomputed, then checked against the reader's flag
    gain = {w: {s: bool(dec[w][s]["delta"] >= bar[w] and dec[w][s]["lo"] > 0) for s in S} for w in W}
    assert all(gain[w][s] == bool(dec[w][s]["gain"]) for w in W for s in S), "gain flag disagrees with n1dw_ho_read.py"
    d1cost = {w: {s: bool(dec[w][s]["cost"]) for s in S} for w in W}
    gcells = {w: {s: set(e1["cost_cells"][f"N1DW{w}_s{s}"]) for s in S} for w in W}
    route_cost = {w: bool(ct["route_cost"][f"N1DW{w}-N1R"]) for w in W}
    rc_rows_ok = {w: all(r.get("status") == "OK" for s in S for r in ct["pairs"][f"N1DW{w}-N1R"][str(s)]["route_setting_RC-A_R_minus_banked"].values()) for w in W}
    guard_holds = {w: (not any(gcells[w][s] for s in S)) and not route_cost[w] for w in W}
    assert all((e1["guard_by_weight"][f"w{w}"]["guard"] == "GUARD-HOLDS") == guard_holds[w] for w in W)
    ng = {w: sum(gain[w].values()) for w in W}
    cost_seeds = {w: sum(bool(gcells[w][s]) for s in S) for w in W}
    lab = {}
    for w in W:
        both_cell = sorted(gcells[w][0] & gcells[w][1])
        rows = {
            "RBW-GAIN": ng[w] == 2 and guard_holds[w],
            "RBW-GAIN-ONE-COST": ng[w] == 2 and cost_seeds[w] == 1,
            "RBW-ONE-SEED": ng[w] == 1 and guard_holds[w],
            "RBW-COSTS": bool(both_cell) or all(d1cost[w].values()),
            "RBW-FLOOR-HIGH": bar[w] > 0.0383 and all(bar[w] > dec[w][s]["delta"] for s in S),
        }
        lab[f"w{w}"] = {"rows": rows, "fired": [k for k, v in rows.items() if v], "gain_seeds": [s for s in S if gain[w][s]], "guard_cost_seeds": [s for s in S if gcells[w][s]],
                        "guard_cost_cells_by_seed": {str(s): sorted(gcells[w][s]) for s in S}, "guard_cost_cell_at_both_seeds": both_cell,
                        "guard_holds": guard_holds[w], "guard_at_gaining_seed_only": all(not gcells[w][s] for s in S if gain[w][s]), "route_cost": route_cost[w],
                        "route_cost_input_complete": rc_rows_ok[w], "bar": r4(bar[w])}
    no_gain_any = all(ng[w] == 0 for w in W)
    dose = no_gain_any and all(bes[4][s]["separated"] and bes[4][s]["delta"] > 0 for s in S) and all(bes[4][s]["delta"] >= bes[2][s]["delta"] for s in S)
    gs2, gs4 = {s for s in S if gain[2][s]}, {s for s in S if gain[4][s]}
    split = all(ng[w] < 2 for w in W) and len(gs2) == 1 and len(gs4) == 1 and gs2 != gs4
    cross = {"RBW-SPLIT": split, "RBW-DOSE": dose, "RBW-NO-GAIN": no_gain_any and not dose}
    reached = sorted({k for w in W for k in lab[f"w{w}"]["fired"]} | {k for k, v in cross.items() if v})
    # four families per weight and seed
    fam = {}
    R1 = d1["R1_LON_LAT_HO_beside_no_bar"]
    g141 = e1["guard_eval141_N1DW-N1R"]
    for w in W:
        for s in S:
            a = f"N1DW{w}_s{s}"
            fam[a] = {"D1-HO_TAC-LAT_N1DW-N1R": ci(dec[w][s]), "D1-HO_TAC-LAT_N1DW-N1D": ci(bes[w][s]),
                      "D1-HO_R1_N1DW-N1D": {c: ci(r) for c, r in R1[a].get("N1DW-N1D", {}).items()},
                      "D1-HO_R1_N1DW-N1R": {c: ci(r) for c, r in R1[a].get("N1DW-N1R", {}).items()},
                      "eval141_N1DW-N1R": {c: ci(r) for c, r in g141[f"{a}-N1R_s{s}"].items()},
                      "eval141_N1DW-N1D": e1["beside_eval141_N1DW-N1D"] if isinstance(e1["beside_eval141_N1DW-N1D"], str) else "see input",
                      "D2-HO_beside": d1["D2_HO_beside"][a], "Z-USE_D1-HO_real-zero": ci(d1["Z_USE_HO_real_minus_zero"][a]),
                      "Z-USE_eval141_RB_tac_lat": e1["Z_USE_eval141_RB_beside"][a]["tac_lat"], "K-CLIP": d1["K-CLIP"][a]}
    strat = {}
    for w in W:
        sv = J[f"s1_w{w}"]
        per = {}
        for s in S:
            b = sv["arms"][f"S1_s{s}"]["block"]
            p = ct["pairs"][f"N1DW{w}-N1R"][str(s)]
            per[str(s)] = {"route_following_level": b["level"], "best_constant_level": b["best_constant"]["level"], "n_by_arc": b["n"],
                           "real_minus_donor": [ci(r) for r in b["real_minus_donor"]], "real_minus_zero": [ci(r) for r in b["real_minus_zero"]],
                           "real_minus_best_constant": [ci(r) for r in b["real_minus_best_constant"]], "minus_C0_real_nav": [ci(r) for r in b["minus_C0_real_nav"]],
                           "N1DW-N1R_S_CKPT": {v: [ci(r) for r in rows] for v, rows in p["S_CKPT_R_minus_banked"].items()},
                           "N1DW-N1R_skill_real-donor": [ci(r) for r in p["skill_real_minus_donor_R_minus_banked"]],
                           "N1DW-N1R_RC-A": {c: ci(r) for c, r in p["route_setting_RC-A_R_minus_banked"].items()},
                           "N1DW-C0_RC-A": {c: ci(r) for c, r in sv["route_setting_RC-A"]["rows"][f"S1_minus_C0_s{s}"].items()},
                           "on_RB_beside": p["on_RB_beside"].get("R_real_minus_donor"),
                           "T1_per_tick_row": sv["arms"][f"S1_s{s}"].get("T1_beside")}
        strat[f"w{w}"] = {"route_following_label_frozen_rule": sv["outcome_route_following"], "per_seed": per, "ROUTE-COST": ct["route_cost"][f"N1DW{w}-N1R"]}
    strat["N1R_reference_label_frozen_rule"] = J["s1_n1r"]["outcome_route_following"]
    strat["K7_K8"] = J["s1_w4"]["controls"]
    nf = J["ctx_nofog_w4"]
    strat["detected_information"] = {"status": "NOT-COMPUTABLE", "reason": "N1DW has no context head: its T0 pass carries no ctx_logits key; strat1_ctx_read.py:6 and its own output 'not_in_arm': " + json.dumps(nf["not_in_arm"]),
                                     "n_heads": 0, "fog_unscored": nf["fog_unscored"]}
    rep = {"tool": "build_verdict.py", "card": "PREREG_V8F_N1DW.md sha256 59f2b7f9 + AMENDMENT_N1DW_1.md sha256 bff20b4d (section 7 as written)",
           "inputs_sha256": {k: sha(p) for k, p in IN.items()}, "inputs": IN,
           "tier": "T1-kin/old (the card's frozen executor) for D1-HO and the eval-141 guard; STRATEGIC T0 at the window's first tick",
           "evidence_class": "MEASURED (ours; existing dumps and reads, zero training)",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0; answers 'another draw of EPISODES' only",
           "labels_by_weight": lab, "cross_weight_rows": cross, "rows_reached": reached,
           "verdict": "NO-ROW-REACHED" if not reached else reached,
           "mean_two_seed_N1DW-N1R": d1["mean_two_seed_N1DW-N1R"], "floor": d1["floor"],
           "controls": {"D1-HO": d1["controls"], "eval141_K-RE": e1["controls"]["K-RE"],
                        "eval141_Z-EVAL": {a: {"pass": z["pass"], "n_windows": z["n_windows"], "max_abs_any_cell": max(c["max_abs_window_diff"] for c in z["cells"].values()),
                                               "nan_mismatch_any_cell": max(c["nan_mismatch"] for c in z["cells"].values())} for a, z in e1["controls"]["Z-EVAL"].items()},
                        "eval141_problems": e1["controls"]["problems"], "strata_141": e1["strata_141"]},
           "families": fam, "STRATEGIC": strat,
           "beside_N1D-N1R_eval141_dk_tg_2_6s_from_RESULT_NAVRBD": {k: ci(v["tactical_2_6s:dk_tg_abs_err_s"]) for k, v in J["navrbd"]["guard_eval141"]["cells_under_F2"].items()},
           "executor_note": "every number is T1-kin/old; the pre-registered overlap read under T1-kin/hb0.2 (PI ruling D16-D, raw/2026-11-19-n1dw/overlap_hb/PREREG_N1DW_HB_OVERLAP.md) is pending and will be reported beside it"}
    json.dump(rep, open(V + "/RESULT_N1DW_VERDICT.json", "w", encoding="utf-8"), indent=1, default=float)
    print("rows reached:", reached or "none")
    for w in W:
        print(f"w{w}", json.dumps(lab[f"w{w}"]))
    print("cross", cross)


if __name__ == "__main__":
    main()
