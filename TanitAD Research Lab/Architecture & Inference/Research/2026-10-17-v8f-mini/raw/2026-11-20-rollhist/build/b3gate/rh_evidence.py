#!/usr/bin/env python3
"""rh_evidence.py -- the evidence file of one ROLLHIST arm for dscale10k_check_rh.py (stdlib only, CPU).

The checker's trainer-side checks (G-HYG, G-LIVE reach table, G-CLOCK, G-EVAL, G-CKPT, G-SUITE) are DECIDED from an evidence file. RH / B0 add no model, loss term, label or clock, so
  * G-HYG / G-LIVE / G-CLOCK / G-EVAL / G-CKPT are INHERITED from the stage-A evidence of the arm's own stage-A model UE_sK (evidence class INHERITED, recorded under ``inherited`` in the file: they were
    measured at cdbff971, the DSCALE10K tree, which is identity to the base 7b26a8ac per the gate; they are NOT re-measured on 2bda9e1a);
  * G-SUITE is FRESH: this commit's G-SUITE result (rh_gsuite.sh: 0 regressions, 0 new-test failures, 0 missing, new commit == 2bda9e1a).
Refuses (rc 1) if the G-SUITE result is not a clean pass at this commit, or the UE evidence lacks a block.
Usage: rh_evidence.py --ue-evidence EVIDENCE_UE_sK.json --gsuite gsuite_result_rh.json --commit SHA40 --out EVIDENCE.json
"""
import argparse
import hashlib
import json
import sys

KEEP = ("hyg", "reach", "clock", "G-EVAL", "G-CKPT")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ue-evidence", required=True)
    ap.add_argument("--gsuite", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ue = json.load(open(a.ue_evidence, encoding="utf-8"))
    raw = open(a.gsuite, "rb").read()
    gs = json.loads(raw)
    bad = [f"the UE evidence has no {k}" for k in KEEP if k not in ue]
    new = gs.get("new_commit", gs.get("new"))
    if new != a.commit:
        bad.append(f"G-SUITE new commit is {new!r}, not {a.commit[:8]}")
    if gs.get("n_regressions") or gs.get("new_test_failures") or gs.get("missing_in_new"):
        bad.append("G-SUITE reads regressions / new-test failures / missing tests")
    if bad:
        for b in bad:
            print("PROBLEM", b)
        return 1
    ev = {k: ue[k] for k in KEEP}
    ev["G-SUITE"] = {"pass": True, "mode": "rollhist-fresh-suite", "n_regressions": 0, "new_commit": a.commit, "suite_sha256": hashlib.sha256(raw).hexdigest()}
    ev["arm_of_inherited"] = ue.get("arm")
    ev["inherited"] = {"blocks": list(KEEP), "evidence_class": "INHERITED (UE stage-A evidence at the DSCALE10K tree, not re-measured on this commit)", "from": a.ue_evidence,
                       "from_sha256": hashlib.sha256(open(a.ue_evidence, "rb").read()).hexdigest()}
    json.dump(ev, open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print("ZZ RH-EVIDENCE WRITTEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
