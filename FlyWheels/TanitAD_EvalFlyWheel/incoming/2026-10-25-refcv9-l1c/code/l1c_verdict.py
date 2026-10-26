"""l1c_verdict.py -- the MECHANICAL verdict of PREREG_L1C.md sections 3-5 from L1C_RESULT.json (+ L1C_ROC.json).  No fitting, no data access.

    python -X utf8 l1c_verdict.py <result_dir>      -> <result_dir>/VERDICT.json + VERDICT_TABLES.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEEDS = ("0", "1")


def sep_below(c):
    return c is not None and c.get("hi") is not None and c["hi"] < 0


def sep_above(c):
    return c is not None and c.get("lo") is not None and c["lo"] > 0


def fmt(c):
    if not c or c.get("delta") is None:
        return "n/a"
    s = " SEP" if (c.get("lo", 0) > 0 or c.get("hi", 0) < 0) else ""
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{s}"


def main() -> int:
    d = Path(sys.argv[1])
    R = json.loads((d / "L1C_RESULT.json").read_text(encoding="utf-8"))
    roc_p = d / "L1C_ROC.json"
    ROC = json.loads(roc_p.read_text(encoding="utf-8")) if roc_p.exists() else None
    S2 = R["surfaces"]["S2"]
    sd = S2["seeds"]
    floor = abs(S2["inference_seed_floor_os"]["delta"])
    K = {}
    K["K1_H0_reproduces"] = all(v["PASS"] for k, v in R["controls_K1"].items() if k.startswith("S2"))
    K["K2_CF2_minus_CF2s_reproduces"] = all(S2["controls"][f"s{s}"]["K2_CF2_minus_CF2s"]["PASS"] for s in SEEDS)
    K["K3_constant_head_R2_le_0"] = all(S2["controls"][f"s{s}"]["K3_PASS"] for s in SEEDS)
    K["K4_fresh_bank_plan_is_roll_os_plan"] = bool(R["bank_controls"]["K4_fresh"]["PASS"])
    K["K4_eval_banks"] = all(R["bank_controls"][f"eval_s{s}"]["PASS"] for s in SEEDS)
    K["K5_fresh_join"] = bool(S2["controls"]["K5_fresh_join"]["PASS"] and S2["controls"]["K5_fresh_join"]["decisions_PASS"]
                              and S2["controls"]["K5_fresh_join"]["validation_PASS"])
    K["V_m_deranged_does_not_beat_H2cs"] = not any(sep_below(sd[s]["paired"]["H2cm-H2cs"]) for s in SEEDS)
    K["arm_identity_g1_zero_prior"] = all(S2["controls"][f"s{s}"]["K2_arm_identity_PASS"] for s in SEEDS)
    if ROC is not None:
        K["ROC_controls"] = bool(ROC.get("controls_all_pass"))
    valid = all(K.values()) and ROC is not None
    P1 = all(sep_below(sd[s]["paired"]["H2c-H2cs"]) and abs(sd[s]["paired"]["H2c-H2cs"]["delta"]) > floor for s in SEEDS)
    P2 = all(sep_below(sd[s]["paired"]["H2c-H0"]) for s in SEEDS)
    worse_both = []
    for k in sd["0"]["four_families"]["H2c-H0"]:
        cells = [sd[s]["four_families"]["H2c-H0"].get(k) for s in SEEDS]
        lower_better = not (k.startswith("TAC_") or k == "LON_speed_within_0p5")
        if all(c is not None and (sep_above(c) if lower_better else sep_below(c)) for c in cells):
            worse_both.append(k)
    auc_read = ROC["readings"].get("L1C_H2c", {}).get("vs_H0") if ROC is not None else None
    auc_down = (auc_read == "AUC-DOWN")
    G = (not worse_both) and (not auc_down)
    dfresh = R["D_fresh"]
    if not valid:
        O = "VOID (a control failed: " + ",".join(k for k, v in K.items() if not v) + (")" if ROC is not None else "; ROC not run)")
    elif P1 and P2 and G:
        O = "O1 SHIP candidate for refcv8 (stage upgrade; needs the registered SHIP-R8 battery on the new stage and the PI's go); refcv9 design confirmed"
    elif P1:
        O = "O2 vision is real on a clean fit but not enough to beat the stage at this head class" + ("" if P2 else " (P2 fails)") + ("" if G else " (G fails)")
    else:
        O = "O3 the cross-fit result does not transfer to a separate calibration set" + (
            ": D-fresh FLAGGED => off-distribution split (a data finding, not a head finding)" if dfresh["flagged"] else
            ": D-fresh not flagged => the CF2 gain was eval-specific; L1 closed for refcv8")
    V = {"controls": K, "valid": bool(valid), "bars": {"P1": P1, "P2": P2, "G": G}, "guard_worse_at_both_seeds": worse_both,
         "braking_auc_H2c_vs_H0": auc_read, "inference_seed_floor_os_m": floor, "branch": R.get("branch"),
         "D_fresh": {"auc": dfresh["domain_auc_compliant_rows"], "flagged": dfresh["flagged"]},
         "D_platform": ({"auc": R["D_platform"].get("domain_auc_rebuilt_vs_shipped"), "confounded_flag": R["D_platform"].get("platform_confounded_flag_auc_gt_0p70")}
                        if "domain_auc_rebuilt_vs_shipped" in R["D_platform"] else R["D_platform"]),
         "outcome": O,
         "estimator_note": "paired episode-cluster bootstrap over episodes, models fixed: answers 'another draw of EPISODES' only; ONE training seed; inference-seed floor applies"}
    (d / "VERDICT.json").write_text(json.dumps(V, indent=1), encoding="utf-8")
    L = [f"# L1C verdict tables ({R['tree']})", "", f"**Outcome: {O}**", "", f"controls {K}", "", f"bars {V['bars']}; braking AUC H2c vs H0: {auc_read}", "",
         f"inference-seed floor (OS s0 vs s1, S2 ADE): {fmt(S2['inference_seed_floor_os'])}", "", f"population {json.dumps(R['population'], default=str)}", "",
         f"D-fresh {json.dumps(V['D_fresh'])}   D-platform {json.dumps(V['D_platform'], default=str)}", ""]
    for surf, SS in R["surfaces"].items():
        L += [f"## {surf}", "", "| arm | E1 s0 | E1 s1 |", "|---|---|---|"]
        for a in SS["seeds"]["0"]["E1_minus_ha0_ext_pose"]:
            L.append(f"| {a} | {fmt(SS['seeds']['0']['E1_minus_ha0_ext_pose'][a])} | {fmt(SS['seeds']['1']['E1_minus_ha0_ext_pose'][a])} |")
        L += ["", "| pair (ADE) | s0 | s1 |", "|---|---|---|"]
        for p in SS["seeds"]["0"]["paired"]:
            L.append(f"| {p} | {fmt(SS['seeds']['0']['paired'][p])} | {fmt(SS['seeds']['1']['paired'][p])} |")
        for p in SS["seeds"]["0"]["four_families"]:
            L += ["", f"four families {p}", "", "| metric | s0 | s1 |", "|---|---|---|"]
            for k in SS["seeds"]["0"]["four_families"][p]:
                L.append(f"| {k} | {fmt(SS['seeds']['0']['four_families'][p][k])} | {fmt(SS['seeds']['1']['four_families'][p][k])} |")
        L += ["", "fit: " + json.dumps({a: {k: v for k, v in f.items() if k in ('k', 'lambda', 'train_R2', 'oof_R2', 'n', 'n_episodes', 'n_params')}
                                         for a, f in SS["fit"].items()}, default=float), "",
              "eval R2 s0: " + json.dumps({a: [round(x, 3) for x in v] for a, v in SS["seeds"]["0"]["eval_R2"].items()}), ""]
        if SS.get("sensitivity"):
            L += ["", "sensitivities (BESIDE, no bar):", "", "| subset | n rows | k | H2c-H2cs s0 | H2c-H2cs s1 | H2c-H0 s0 | H2c-H0 s1 |", "|---|---|---|---|---|---|---|"]
            for nm, v in SS["sensitivity"].items():
                if "skipped" in v:
                    L.append(f"| {nm} | skipped: {v['skipped']} | | | | | |")
                else:
                    L.append(f"| {nm} | {v['n_rows']} | {v['k']} | {fmt(v['seeds']['0']['H2c-H2cs'])} | {fmt(v['seeds']['1']['H2c-H2cs'])} | "
                             f"{fmt(v['seeds']['0']['H2c-H0'])} | {fmt(v['seeds']['1']['H2c-H0'])} |")
    (d / "VERDICT_TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[l1c-verdict] {O}  ZZL1CVERDICTZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
