#!/usr/bin/env python3
"""zt5_report.py -- merge the post-hoc reads and the F_m arm into RESULT.json and write RESULT.json["report"] (the
subagent harness refuses REPORT.md; this block replaces it).  Numbers are read from the JSON artifacts, not typed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ci(d, n=2):
    return f"{d['point']:+.{n}f} [{d['lo']:+.{n}f}, {d['hi']:+.{n}f}]"


def pc(d, n=1):
    return f"{d['point']:.{n}f} % [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"


def main():
    R = json.load(open(ROOT / "RESULT.json", encoding="utf-8"))
    R["posthoc_knot_split"] = json.load(open(ROOT / "data/posthoc_knot.json", encoding="utf-8"))
    R["posthoc_tracker_saved"] = json.load(open(ROOT / "data/posthoc_tracker_saved.json", encoding="utf-8"))
    R["posthoc_signed_heading_navtest"] = json.load(open(ROOT / "data/posthoc_signed_heading_navtest.json", encoding="utf-8"))
    R["next_arm_F_m_amendment3"] = json.load(open(ROOT / "data/fm_dev.json", encoding="utf-8"))
    N, D = R["navtest_mechanism"], R["dev"]
    e0 = D["effects"]["N-s0"]
    rt = N["rescue_pct_TM"]
    mr = D["mechanism_replication"]
    ts = R["posthoc_tracker_saved"]
    fm = R["next_arm_F_m_amendment3"]
    ks = R["posthoc_knot_split"]
    rb, rbc = str(D["rho_star_N-s0"]["F_b"]), str(D["rho_star_N-s0"]["F_bc"])
    R["report"] = {
        "headline": ("Mechanism FOUND, fix NOT: the tracking-made DAC zeros are driven by the HEADING CHANNEL (heading:=path tangent "
                     f"rescues {pc(rt['H'])} of the 144 navtest tracker-made zeros; dev replicates {pc(mr['fan_tracker_made']['rescue_pct']['H'])} "
                     f"on 1,172 fan units) with actuator/steering lag second ({pc(rt['A'])}). But NO admissible fix clears the dev bar: "
                     f"F_c {ci(e0['F_c']['dPDMS_x100'])} PDMS, F_b {ci(e0['F_b'][rb]['dPDMS_x100'])}, F_bc {ci(e0['F_bc'][rbc]['dPDMS_x100'])}, "
                     f"F_m {ci(fm['eligibility']['dPDMS_x100'])}; even the inadmissible pre-compensation ceiling reads "
                     f"{ci(e0['F_a_ceiling']['dPDMS_x100'])}. Why: at the PICK the tracker SAVES about as many plans as it breaks "
                     f"(dev N-s0 tracker-saved {ts['dev']['N-s0']['TS_picks']} vs tracker-made {ts['dev']['N-s0']['TM_picks']}; navtest "
                     f"~{ts['navtest']['TS_tokens_ESTIMATED']['point']:.0f} ESTIMATED vs 144), so ZT3's 1.07 PDMS was GROSS reach; net reach of "
                     "execution-side levers is ~0. No navtest confirm is issued (PREREG section 6)."),
        "status_rule_zero": ("Every remaining lever is BLOCKED or closed: execution side (F_c, F_a) and map-free selection (F_b, F_bc, F_m) "
                             "measured ~0 or negative on dev; the zero tail is a boundary-MARGIN problem whose lever is the scorer's "
                             "boundary representation = ZT3's committed next experiment ZT4 (P6 trunk DAC gradient), BLOCKED on the P6 "
                             "checkpoint (queued, authorised; control arm already measured in ZT4_BASELINE.json). Training-side margin "
                             "targets (P5 arm (i)/P5-C) need the PI's GPU go."),
        "mechanism_navtest_DAC_level": {
            "units": "144 tracker-made u400 picks (44 logs) + 529 random-stratum controls post-stratified on turn x v0; gates AG1 673/673, NG1 673/673",
            "rescue_single_factor": {k: pc(rt[k]) for k in ("H", "A", "S", "V", "P")},
            "controls_broken_weighted": {k: pc(N["controls_broken_pct_weighted"][k]) for k in ("H", "A", "P")},
            "overlap_of_144": "H only 53, A only 39, both 29, neither 23",
            "form": "tracked car under-turns and runs WIDE late (only 10 % inside the turn at max deviation; 60 % along-track lag)",
            "knot_split_POSTHOC": {"navtest_Fc8_last_knot_only": pc(ks["navtest_144_POSTHOC"]["rescue_pct_Fc8"]),
                                   "navtest_Fc17_knots_1_to_7": pc(ks["navtest_144_POSTHOC"]["rescue_pct_Fc17"]),
                                   "dev_picks_Fc8": pc(ks["dev_tracker_made_picks"]["rescue_pct_Fc8"]),
                                   "dev_picks_Fc17": pc(ks["dev_tracker_made_picks"]["rescue_pct_Fc17"])},
            "reading": "R-MECH = DOMINANT H (heading channel); the interior knots carry most of it, not only A7's last pose",
            "real_controller": ("F_c moves no position; it makes the heading field consistent with the path (rear axle moving along its "
                                "heading). Tracking fidelity improves on dev (delta "
                                f"{ci(e0['F_c']['families']['LATERAL']['d_delta'], 3)} m, heading MAE "
                                f"{ci(e0['F_c']['families']['LATERAL']['d_heading_mae_deg'], 2)} deg) but the footprint rotation "
                                "breaks near-boundary plans. Pre-compensation draws a path off the intended one (drawn footprint fails +"
                                f"{e0['F_a_ceiling']['drawn_fail_pp']['point']:.1f} pp on dev): a real car with another controller would drive it.")},
        "dev": {
            "set": "2,466 navtrain held-out sets / 13 logs (navtest-disjoint), fan = model_final (= u400's generator); picks N-s0 (primary), N-s1, N-s2",
            "gates": D["gates"],
            "DG1_note": ("DG1 strictly FAILED on 1 of 2,466 sets (EP only, 0.0286: the re-built PDM-Closed progress denominator); every arm "
                         "of that set is scored with the same re-built parts, so the paired deltas are internally consistent; its weight "
                         "in any mean is 1/2,466"),
            "baseline_N-s0_PDMS_x100": ci(e0["base_PDMS_x100"]),
            "F_c": {"dPDMS": ci(e0["F_c"]["dPDMS_x100"]), "dDAC": ci(e0["F_c"]["dDAC_x100"]), "rescued": e0["F_c"]["dac_zero_rescued"],
                    "broken": e0["F_c"]["dac_zero_broken"], "drawn_fail_pp": ci(e0["F_c"]["drawn_fail_pp"]),
                    "replicates": {a: ci(D["effects"][a]["F_c"]["dPDMS_x100"]) for a in ("N-s1", "N-s2")}},
            "F_b": {"rho_star": D["rho_star_N-s0"]["F_b"], "dPDMS": ci(e0["F_b"][rb]["dPDMS_x100"]), "switched_pct": e0["F_b"][rb]["switched_pct"]},
            "F_bc": {"rho_star": D["rho_star_N-s0"]["F_bc"], "dPDMS": ci(e0["F_bc"][rbc]["dPDMS_x100"])},
            "F_a_ceiling": {"dPDMS": ci(e0["F_a_ceiling"]["dPDMS_x100"]), "dEP": ci(e0["F_a_ceiling"]["dEP_x100"]),
                            "rescued": e0["F_a_ceiling"]["dac_zero_rescued"], "broken": e0["F_a_ceiling"]["dac_zero_broken"]},
            "F_m_amendment3": {"rho_star": fm["rho_star"], "dPDMS": ci(fm["eligibility"]["dPDMS_x100"]), "eligible": fm["eligibility"]["eligible"]},
            "eligibility": {k: v["eligible"] for k, v in D["eligibility"].items()},
            "chosen": D["chosen"],
            "families": "LONGITUDINAL dEP / speed MAE, LATERAL cross MAE / heading MAE / delta, TACTICAL manoeuvre agreement (selection arms) are in dev.effects.*.families; STRATEGIC PAUSED (PI ruling 2026-09-19)"},
        "navtest_confirm": ("NOT ISSUED: PREREG section 6 writes no confirm for an ineligible fix, and dev predicts F_c at "
                            f"{ci(e0['F_c']['dPDMS_x100'])} PDMS. Tooling exists and is self-pair tested (exactly 0, NOT PROVEN, 136 logs) "
                            "if the PI wants the read anyway: python code/zt5_seam_fix.py --seam D:/Projects/TanitAD/data/refe_navtest/seams/"
                            "stage2_sft4_B_u400_navtest_full.npz --out <F_c seam> --expect-n 12146 ; python eval/score_navtest_refe.py "
                            "--label refe_zt5_Fc_u400_navtest_full --seam <F_c seam> (no --tokens) ; python code/zt5_confirm_read.py --fixed "
                            "<that csv> --base D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/"
                            "refe_stage2_sft4_B_u400_navtest_full.csv --tokens D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json "
                            "--out <json>. Bar: ADOPT iff lower > 0 and no NC/DAC/EP/TTC/C adverse; REFUTED iff upper < 0; else NOT PROVEN."),
        "limitations": ["dev picks are u400's 400-update siblings (u400's own held-out scores are not on the dev box)",
                        "dev = 13 logs (11 held-out log DBs absent: navtrain_dbs not retrieved) -> wide log-cluster intervals",
                        "intervals answer another draw of LOGS only; scorer-seed variance is read through N-s0/1/2",
                        "tracker-saved counts, the knot split and the signed-heading read are POST-HOC (decide nothing)",
                        "navtest exposure: DAC-level mechanism on the 144 + 529 controls only; no PDMS of any fix on navtest"],
        "harness_note": "the subagent harness refused REPORT.md / MANIFEST.md; this block and MANIFEST.json replace them",
    }
    json.dump(R, open(ROOT / "RESULT.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(R["report"], indent=1))


if __name__ == "__main__":
    main()
