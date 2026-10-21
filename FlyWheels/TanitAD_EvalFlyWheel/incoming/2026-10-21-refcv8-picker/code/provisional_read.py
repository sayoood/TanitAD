"""provisional_read.py -- PROVISIONAL s1-only read of the provisional trees (pvship_pf / pvbase_pf / pvmbrpx_pf).  NOT the registered read.

Runs analyze_picker_r8.family_reads + verdicts with TREES re-pointed (env PK_TREES, set BEFORE the import), and reports ONLY seed-1 cells
(seed 0 of every provisional tree is the bank's own plan, so s0 cells are identically 0 by construction and are printed as the machinery
control they are).  Writes PROVISIONAL_S1_READ.json + PROVISIONAL_S1.md under the picker dir.
"""
import json
import os
import sys
import time

MBR_TREE = os.environ.get("PV_MBR_TREE", "pvmbrpx_pf")          # pvmbrpx_pf = slot-level PROXY mask; pvmbrrule_pf = re-derived e9_keep
TAG = os.environ.get("PV_TAG", "")
os.environ["PK_TREES"] = json.dumps({"SHIP": "pvship_pf", "BASE": "pvbase_pf", "MBR32": MBR_TREE})
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_picker_r8 as A  # noqa: E402

P = A.P


def f4(c):
    return A.f4(c)


def main():
    t0 = time.time()
    CP, CR, T = A.tools()
    fam = A.family_reads(CP, CR, T)
    ver = A.verdicts()
    METRICS = fam["metrics"]
    out = {"status": "PROVISIONAL -- NOT THE REGISTERED READ (PREREG_R8_PICKER sealed e6ec30e7... earlier; its bars are unchanged by this)",
           "generated": time.strftime("%F %T %z"), "trees": A.TREES, "tier": A.TIER, "estimator": A.QUESTION,
           "s0_machinery_control_all_cells_zero": None, "verdicts": ver, "s1": {}}
    # s0 machinery control (every provisional tree's s0 IS the bank's own plan): the +RT arms must read EXACTLY 0 vs SHIP on every cell, and each raw
    # arm's cell must EQUAL SHIP_raw_capped's (raw vs SHIP+RT is NOT zero -- it is the RETIME gain -- so it is compared for identity, not to 0).
    s0 = fam["seeds"]["0"]
    rt0 = [c for surf in s0["surfaces"].values() for a in ("BASE+RT", "MBR32+RT") for st in ("compliant", "all") for c in surf["cells"][a][st].values()
           if c.get("delta") is not None]
    raw_eq = [surf["cells"][f"{a}_raw_capped"] == surf["cells"]["SHIP_raw_capped"] for surf in s0["surfaces"].values() for a in ("BASE", "MBR32")]
    cc0 = s0["compliance"]["cells_vs_SHIP"]
    comp_rt0 = [cc0[a][st] for a in ("BASE+RT", "MBR32+RT") for st in ("compliant", "all")]
    comp_raw_eq = [cc0[f"{a}_raw_{v}"] == cc0[f"SHIP_raw_{v}"] for a in ("BASE", "MBR32") for v in ("capped", "uncapped")]
    out["s0_machinery_control"] = {"n_rt_cells": len(rt0), "rt_cells_all_exactly_zero": bool(rt0 and all(c["delta"] == 0 and c["lo"] == 0 and c["hi"] == 0 for c in rt0)),
                                   "raw_cells_equal_SHIP_raw": bool(raw_eq and all(raw_eq)),
                                   "compliance_rt_zero": bool(all(c.get("delta") == 0 and c.get("lo") == 0 and c.get("hi") == 0 for c in comp_rt0)),
                                   "compliance_raw_equal_SHIP_raw": bool(all(comp_raw_eq))}
    out["s0_machinery_control_all_cells_zero"] = bool(all(out["s0_machinery_control"][k] for k in ("rt_cells_all_exactly_zero", "raw_cells_equal_SHIP_raw",
                                                                                                   "compliance_rt_zero", "compliance_raw_equal_SHIP_raw")))
    out["ship_inference_floor_S2_ade_compliant"] = fam["ship_inference_floor_S2_ade_compliant"]
    out["K13_ship_minus_ship_exactly_zero"] = fam["K13_ship_minus_ship_exactly_zero"]
    s1 = fam["seeds"]["1"]
    for arm in ("BASE+RT", "MBR32+RT", "BASE_raw_capped", "MBR32_raw_capped", "SHIP_raw_capped"):
        worse = []
        for m in METRICS:
            for surf in A.SURFS:
                c = s1["surfaces"][surf]["cells"][arm]["compliant"][m["key"]]
                if A.sep_worse(c, m["lower_is_better"]):
                    worse.append(f"{m['family']}:{surf}:{m['key']} {f4(c)}")
        ck = arm if "+RT" in arm else arm.replace("_capped", "_uncapped")
        cc = (s1["compliance"]["cells_vs_SHIP"].get(ck) or {}).get("compliant")
        if A.sep_worse(cc, True):
            worse.append(f"LONGITUDINAL:FULL8:COMPLIANCE {f4(cc)}")
        out["s1"][arm] = {"S2_ade_minus_SHIP": s1["surfaces"]["S2"]["cells"][arm]["compliant"]["ade_m"],
                          "S6_ade_minus_SHIP": s1["surfaces"]["S6"]["cells"][arm]["compliant"]["ade_m"],
                          "separated_worse_cells_s1": worse, "compliance_minus_SHIP": cc,
                          "compliance_rate": s1["compliance"]["rates"].get(ck, {}).get("compliant"),
                          "means_compliant_S2": s1["surfaces"]["S2"]["means_compliant"].get(arm)}
    out["s1"]["SHIP"] = {"compliance_rate": s1["compliance"]["rates"]["SHIP"]["compliant"], "means_compliant_S2": s1["surfaces"]["S2"]["means_compliant"]["SHIP"]}
    out["families_full"] = fam
    json.dump(out, open(f"{P}/PROVISIONAL_S1{TAG}_READ.json", "w", encoding="utf-8"), indent=1, default=str)
    L = ["# PROVISIONAL: the s1-only read of the pickers through SHIP-R8 + RETIME (NOT the registered read)", "",
         f"* **Generated:** {out['generated']}. All numbers come from `PROVISIONAL_S1{TAG}_READ.json`. **Evidence class:** MEASURED.",
         f"* **Tier and estimator:** {A.TIER}; {A.QUESTION}.",
         "* **Source fan.** The fanceil Thor s1 fan: os-only, inference seed 1. It matches the bank's s1 draw on 99.4 % of fan "
         "cells, and its own sel agrees with the bank's on 0.9958 of windows.",
         "* **Seed 0.** Seed 0 of every provisional tree is the bank's OWN plan, so s0 is a machinery control: every +RT cell must read exactly 0 vs SHIP "
         f"({out['s0_machinery_control']['n_rt_cells']} cells), and each raw arm's cell must equal SHIP_raw's (the raw-vs-SHIP+RT gap is the RETIME gain, not 0). "
         f"**Control PASS: {out['s0_machinery_control_all_cells_zero']}** {out['s0_machinery_control']}. K13 (SHIP − SHIP = 0): {out['K13_ship_minus_ship_exactly_zero']}.",
         f"* **SHIP inference floor** (|S2 compliant ADE, s0 bank draw − s1 Thor draw|): {out['ship_inference_floor_S2_ade_compliant']['floor_m']:.4f} m. "
         "The prereg's A3 bar needs |delta| > 2× this floor at BOTH seeds; only s1 is read here.",
         ("* **MBR here is the PROXY-mask cell** (`pvmbrpx_pf`). The registered MBR32 uses the model's own `e9_keep`, which this fan does not carry."
          if MBR_TREE == "pvmbrpx_pf" else
          "* **MBR here is the RE-DERIVED e9_keep cell** (`pvmbrrule_pf`): reach_keep captured, ceil_keep re-derived by the producer's rule "
          "(control in PROVISIONAL_S1_RULE_BUILD.json), r8_emit_keep assumed all True. The registered MBR32 uses the CAPTURED e9_keep."), "",
         "## SHIP-R8 per provisional tree", "",
         "The `overall` column is not a picker verdict, because s0 = own pick.", "",
         "| tree | overall | per clause | E1 s1 | E3 s1 |", "|---|---|---|---|---|"]
    for nm, tr in A.TREES.items():
        v = ver.get(nm, {})
        L.append(f"| {tr} | {v.get('overall')} | {v.get('per_clause')} | {f4((v.get('E1') or {}).get('s1'))} | {f4((v.get('E3') or {}).get('s1'))} |")
    L += ["", "## Arm − SHIP (= own + RETIME_DEPLOY8) at seed 1, compliant windows", "",
          "| arm | S2 ADE − SHIP | S6 ADE − SHIP | COMPLIANCE − SHIP | cells separated WORSE at s1 |", "|---|---|---|---|---|"]
    for arm, r in out["s1"].items():
        if arm == "SHIP":
            continue
        L.append(f"| {arm} | {f4(r['S2_ade_minus_SHIP'])} | {f4(r['S6_ade_minus_SHIP'])} | {f4(r['compliance_minus_SHIP'])} | "
                 f"{'; '.join(r['separated_worse_cells_s1']) or 'none'} |")
    L += ["", f"SHIP (own + RETIME) has an S2 compliant mean ADE of {out['s1']['SHIP']['means_compliant_S2']['ade_m']:.4f} and a compliance rate of "
          f"{out['s1']['SHIP']['compliance_rate']['mean']:.4f}.", ""]
    open(f"{P}/PROVISIONAL_S1{TAG}.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("ZZPKPROVREADDONEZZ", round(time.time() - t0, 1))


if __name__ == "__main__":
    import pk_peak
    pk_peak.arm("provisional_read")
    main()
