"""D6 verification -- re-derive the headline numbers of RESULT.md by a path that does NOT go through d6_common / the part scripts:
the official score CSVs (stage_one / stage_two columns), the PRE_AGGREGATION frames, and the raw rescore JSONL.  Includes a
mutation control (a deliberately wrong expectation must be REJECTED by the same comparison).
    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe d6_verify.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
PKG = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw"
S30 = f"{PKG}/milestones/step30000/scores_navhard"
checks = []


def check(name, got, want, tol=0):
    ok = abs(got - want) <= tol
    checks.append({"name": name, "got": got, "want": want, "ok": bool(ok)})
    return ok


def main():
    csv = pd.read_csv(f"{S30}/score_R7_A1__navhard_two_stage.csv")
    tok = csv[~csv.token.str.startswith("extended_pdm_score")]
    s1 = tok[tok.drivable_area_compliance_stage_one.notna()]
    s2 = tok[tok.drivable_area_compliance_stage_two.notna()]
    check("csv rows stage1", len(s1), 450)
    check("csv rows stage2", len(s2), 5462)
    check("A1 DAC zero stage1 (official CSV)", int((s1.drivable_area_compliance_stage_one == 0).sum()), 145)
    check("A1 DAC zero stage2 (official CSV)", int((s2.drivable_area_compliance_stage_two == 0).sum()), 1418)
    check("A1 NC zero stage1 (official CSV)", int((s1.no_at_fault_collisions_stage_one == 0).sum()), 35)
    check("A1 NC zero stage2 (official CSV)", int((s2.no_at_fault_collisions_stage_two == 0).sum()), 854)
    pre = pd.read_csv(f"{S30}/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_final_scores_frame_PRE_AGGREGATION.csv")
    check("PRE_AGGREGATION DAC zeros (both stages)", int((pre.drivable_area_compliance == 0).sum()), 1563)
    check("PRE_AGGREGATION NC zeros (both stages)", int((pre.no_at_fault_collisions == 0).sum()), 889)
    s1csv = pd.read_csv(f"{S30}/score_R7_A1_s1__navhard_two_stage_wrapper/R7_A1_s1__navhard_two_stage_final_scores_frame_PRE_AGGREGATION.csv").set_index("token")
    pre = pre.set_index("token")
    z = pre[pre.drivable_area_compliance == 0].index
    check("seed-1 plan DAC-clean on A1's DAC-zero scenes", int((s1csv.loc[z, "drivable_area_compliance"] != 0).sum()), 121)
    # raw rescore jsonl
    n_ref_fail = n_nd0 = n_ok = 0
    max_repro = 0.0
    A = {}
    for l in open(f"{RAW}/rescore_A_dac0.jsonl", encoding="utf-8"):
        r = json.loads(l)
        A[r["token"]] = r
        n_ok += r["status"] == "OK"
        max_repro = max(max_repro, max(r["repro"].values()))
        n_ref_fail += r["ref_dac"] == 0
        n_nd0 += r["nd_first"] == 0
    check("rescore A tokens OK", n_ok, 1563)
    check("rescore A max abs diff to banked sub-scores", max_repro, 0.0)
    check("DAC-zero scenes whose reference also fails DAC", n_ref_fail, 427)
    check("DAC-zero scenes non-drivable at t=0", n_nd0, 0)
    check("rescore A tokens == official DAC-zero set", int(set(A) == set(z)), 1)
    # NC front collisions among NC-zero (from both shards)
    nz = set(pre[pre.no_at_fault_collisions == 0].index)
    front = 0
    seen = 0
    for fn in ("rescore_A_dac0.jsonl", "rescore_B_nc_ddc_ctrl.jsonl"):
        for l in open(f"{RAW}/{fn}", encoding="utf-8"):
            r = json.loads(l)
            if r["token"] in nz:
                seen += 1
                ev = [e for e in r["nc_events"] if e["at_fault"]]
                if ev and ev[0]["t_idx"] > 1 and ev[0]["ctype"] in ("ACTIVE_FRONT_COLLISION", "STOPPED_TRACK_COLLISION"):
                    front += 1
    check("NC-zero scenes re-scored", seen, 889)
    check("NC-zero scenes whose first at-fault event is a front collision (t>0.1 s)", front, 843)
    # mutation control: the same comparison must reject a wrong expectation
    mut = {"name": "MUTATION: wrong expectation (DAC zero stage2 = 1417) must be rejected", "got": int((s2.drivable_area_compliance_stage_two == 0).sum()), "want": 1417}
    mut["ok"] = abs(mut["got"] - mut["want"]) > 0
    checks.append(mut)
    out = {"n_checks": len(checks), "n_ok": sum(c["ok"] for c in checks), "checks": checks}
    json.dump(out, open(f"{RAW}/d6_verify.json", "w", encoding="utf-8"), indent=1)
    for c in checks:
        print(("OK  " if c["ok"] else "FAIL"), c["name"], c["got"], c["want"])
    print(out["n_ok"], "/", out["n_checks"])


if __name__ == "__main__":
    main()
