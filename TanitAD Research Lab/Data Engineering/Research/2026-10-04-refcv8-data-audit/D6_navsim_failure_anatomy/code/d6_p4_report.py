"""D6 P4 -- render the generated tables of RESULT_P4.md from raw/p4_result.json (do not hand-edit the output).   tanitad venv.

    python d6_p4_report.py   -> raw/p4_tables.md
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
NAMES = {"G0": "G0 (deployed pick)", "G1": "G1 (own-map gate)", "G1DER": "G1-DER (deranged mask)", "GORC": "G-ORC (GT drivable, PRIVILEGED)"}


def pp(x):
    return f"{100 * x:+.2f}"


def ci(c, scale=1.0, fmt="{:+.4f}"):
    return "[" + ", ".join(fmt.format(scale * v) for v in c) + "]"


def main():
    r = json.load(open(os.path.join(RAW, "p4_result.json"), encoding="utf-8"))
    A = r["arms"]
    L = ["### P4-T1 Official navhard two-stage EPDMS and zero-rates per arm (refcv7 step 50,400, inference seed 0; all 5,912 tokens; MEASURED, local devkit)", "",
         "| arm | EPDMS (devkit) | dEPDMS vs G0 [95 % paired log-cluster CI] | DAC0 | dDAC0 pp [CI] | NC0 | dNC0 pp [CI] | EP mean | dEP [CI] | changed picks | unchecked points (picked plan / all candidates) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ("G0", "G1", "G1DER", "GORC"):
        if a not in A:
            continue
        x = A[a]
        if a == "G0":
            d = dd = dn = de = "--"
            unc = "--"
        else:
            d = f"{x['dEPDMS_vs_G0']['delta']:+.4f} {ci(x['dEPDMS_vs_G0']['ci95'])}"
            dd = f"{pp(x['dDAC0_vs_G0']['delta'])} {ci(x['dDAC0_vs_G0']['ci95'], 100, '{:+.2f}')}"
            dn = f"{pp(x['dNC0_vs_G0']['delta'])} {ci(x['dNC0_vs_G0']['ci95'], 100, '{:+.2f}')}"
            de = f"{x['dEP_mean_vs_G0']['delta']:+.4f} {ci(x['dEP_mean_vs_G0']['ci95'])}"
            g = x["gate_summary"]
            unc = f"{100 * g['unchecked_share_picked_plan_points']:.1f} % / {100 * g['unchecked_share_all_candidates_points']:.1f} %"
        L.append(f"| {NAMES[a]} | {x['EPDMS']:.4f} | {d} | {100 * x['DAC0_rate']:.2f} % | {dd} | {100 * x['NC0_rate']:.2f} % | {dn} | "
                 f"{x['EP_mean_tokens']:.4f} | {de} | {x['n_changed']} ({100 * x['changed_share']:.1f} %) | {unc} |")
    sf = r["seed_floor"]
    L += ["", f"Seed floor (banked R7_A1 vs R7_A1_s1, same tokens): |dEPDMS| = {sf['abs_delta']:.4f} (2x = {sf['two_x']:.4f}); |dDAC0| = {100 * sf['DAC0_abs_delta']:.2f} pp.", ""]
    L += ["### P4-T2 Per stratum (P1' rule sets; per-token means; dDAC0 vs G0 paired log-cluster CI)", "",
          "| arm | stratum | n | DAC0 | dDAC0 pp [CI] | NC0 | EP mean | per-token score mean | changed share |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ("G0", "G1", "G1DER", "GORC"):
        if a not in A:
            continue
        for s, v in A[a]["strata"].items():
            d = "--" if a == "G0" else f"{pp(v['dDAC0_vs_G0']['delta'])} {ci(v['dDAC0_vs_G0']['ci95'], 100, '{:+.2f}')}"
            L.append(f"| {NAMES[a]} | {s} | {v['n']} | {100 * v['DAC0']:.2f} % | {d} | {100 * v['NC0']:.2f} % | {v['EP_mean']:.4f} | "
                     f"{v['score_mean']:.4f} | {100 * v['changed_share']:.1f} % |")
    L += ["", "### P4-T3 Gate statistics", "", "| arm | deployed pick passes | fallback (no reach-kept candidate passes) | of which some candidate passes outside reach_keep |",
          "|---|---|---|---|"]
    for a in ("G1", "G1DER", "GORC"):
        if a in A:
            g = A[a]["gate_summary"]
            L.append(f"| {NAMES[a]} | {100 * g['deployed_pick_passes_share']:.1f} % | {g['n_fallback_none_pass']} | {g['n_fallback_but_some_pass_outside_reach_keep']} |")
    v = r["verdict"]
    L += ["", "### P4-T4 Registered clauses (SPEC s4 literals)", "", "```", json.dumps(v, indent=1), "```"]
    open(os.path.join(RAW, "p4_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
