#!/usr/bin/env python3
"""Write eval/rls1d_arms.json: R0's banked token argv (derived from eval/rls1c_arms.json exactly as that file was derived) + the SAFEOBJ levers.
usage: make_arms_json.py <PK dir>      (writes <PK>/eval/rls1d_arms.json; prints the argv sha256 of each arm and the R0 self-check)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import patchlib as PL  # noqa: E402

R0_SHA = "24451b621d515066bf8febefaf66bea2c7a345b48bf7a89c9443e2d848d8eef6"
CON_ON = ["--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0"]
CON_OFF = ["--epo-con", "off"]


def main():
    pk = PL.pk_arg()
    sys.path.insert(0, os.path.join(pk, "refe"))
    import gate_token as GT  # noqa: E402
    c = json.load(open(os.path.join(pk, "eval", "rls1c_arms.json"), encoding="utf-8"))
    r0c = c["arms"]["R0c"]["argv"]
    assert r0c[-2:] == ["--epo-ref", "armA"], r0c[-4:]

    def base(arm, seed):
        a = [t.replace("/arms/R0c", f"/arms/{arm}") for t in r0c[:-2]]
        i = a.index("--seed")
        a[i + 1] = str(seed)
        return a
    r0 = base("R0", 0)
    assert GT.argv_sha256(r0) == R0_SHA, ("BASE with ARM=R0, S=0 does not hash to R0's banked token", GT.argv_sha256(r0))
    arms, out = {}, {}
    for arm, seed, lever, cls_note in (("R0d", 0, CON_ON, "its OWN arm A (same pass); C0 (banked) is the SECOND control"),
                                       ("R1d", 1, CON_ON, "its OWN arm A (same pass); R1o is its matched lever-off control on identical trainer bytes"),
                                       ("R1o", 1, CON_OFF, "its OWN arm A; the matched lever-off control of R1d (K3 OBSERVE-ONLY, K2/K6 live)")):
        argv = base(arm, seed) + lever
        arms[arm] = {"seed": seed, "class": "R", "control": cls_note, "argv": argv, "args": " ".join(argv), "argv_sha256": GT.argv_sha256(argv)}
        out[arm] = arms[arm]["argv_sha256"][:12]
    doc = {"schema": "rls1d_arms/1",
           "prereg": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md (frozen at 19785c8; running R0d is a PI + Master Mind decision)",
           "profile": "refe_rls1d",
           "lever": "--epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0 (constrained EPO: TTC and NC against arm A, projected dual ascent); R1o names --epo-con off; everything else is R0's argv token for token",
           "derived_from": {"arm": "R0", "token": "HF Sayood/tanitad-refe-rls1-r0 final/PASS_a120418a870f.json", "argv_sha256": R0_SHA,
                            "commit": "a120418a870f94f7f769b24a593ea0c91d131c3d", "via": "eval/rls1c_arms.json (R0c minus --epo-ref armA, /arms/R0c -> /arms/R0)"},
           "order": ["R0d", "R1d", "R1o"],
           "stop_rules": "a K3 kill of R0d or R1d STOPS the chain (generic killed-arm refusal); R1o runs only if R0d and R1d both read bar S = PASS (rls1_checks.py safeobj-gate); R1o runs K3 OBSERVE-ONLY",
           "arms": arms,
           "self_check": {"R0_argv_rebuilt_sha256": GT.argv_sha256(r0), "equals_R0_banked_token": GT.argv_sha256(r0) == R0_SHA,
                          "lever_tokens_not_in_R0": {a: [t for t in arms[a]["argv"] if t not in r0] for a in arms},
                          "note": "paths are the pod chain's ($R = /workspace/data/rls1); BASE = R0c's argv with the lever removed and /arms/R0c swapped back, which hashes to R0's banked token"},
           "budget": {"a40_arms": 3, "hours_per_arm_planned": 3.5, "total_a40_hours": 10.5,
                      "basis": "R0 MEASURED 3.4 A40-h (raw/2026-10-14-rls1-r0-kill/README.md:44); overhead of the constraint ESTIMATED < 1 % (SPEC_SAFEOBJ section 8); Stage 2 (R1d + R1o) is a PI decision"}}
    path = os.path.join(pk, "eval", "rls1d_arms.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(json.dumps({"wrote": path, "argv_sha256_12": out, "R0_selfcheck": GT.argv_sha256(r0)[:12]}))


if __name__ == "__main__":
    main()
