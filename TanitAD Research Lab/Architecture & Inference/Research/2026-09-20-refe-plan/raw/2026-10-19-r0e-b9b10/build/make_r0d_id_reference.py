#!/usr/bin/env python3
"""Write eval/r0d_id_reference.json: the numbers PREREG_R0E_PICK section 5.1 compares R0e-id against, taken from R0d's BANKED outcome files and from nothing else.

usage: make_r0d_id_reference.py <PK dir>     (writes <PK>/eval/r0d_id_reference.json, LF; reads <PK>/raw/2026-10-18-r0d-outcome/raw/{sft.jsonl,rlval_B_vs_ownA_u250.json})

Why a file: the reader (eval/r0e_id_read.py) runs on the pod, and the pod package does not carry raw/2026-10-18-r0d-outcome/.  The values are therefore FROZEN here with the sha256 of the banked
file each came from; eval/test_r0e_b9b10 re-derives them from the banked files whenever those are present (a reference that disagrees with its source is a RED test, not a quiet drift).

  data event      epo_con (the eight keys), epo (the whole block), train_sets            <- sft.jsonl, event "data"          (PREREG 5.1 ID-FAIL clauses 1-3)
  step0 eval      base.pick / A.pick / B.pick                                            <- sft.jsonl, event "eval" tag step0 (PREREG 5.1 clause 6; the prereg quotes base.pick 90.8428)
  u250 contrast   B - own A, TTC and PDMS, x100                                          <- rlval_B_vs_ownA_u250.json paired_vs_base.B.quantities.{TTC,PDMS}.mean (PREREG 5.1 ID-DIVERGED: TTC -1.167, PDMS +0.683)
"""
import hashlib
import json
import os
import sys


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def main():
    pk = os.path.abspath(sys.argv[1])
    base = os.path.join(pk, "raw", "2026-10-18-r0d-outcome", "raw")
    sft = os.path.join(base, "sft.jsonl")
    u250 = os.path.join(base, "rlval_B_vs_ownA_u250.json")
    rows = [json.loads(ln) for ln in open(sft, encoding="utf-8") if ln.strip()]
    data = [r for r in rows if r.get("event") == "data"]
    s0 = [r for r in rows if r.get("event") == "eval" and r.get("tag") == "step0"]
    assert len(data) == 1 and len(s0) == 1, (len(data), len(s0))
    d, e = data[0], s0[0]
    assert sorted(d["epo_con"]) == sorted(["mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref"]), sorted(d["epo_con"])        # "R0d's eight keys"
    assert "temp" not in d["epo_con"] and d["train_sets"] == 16000
    q = json.load(open(u250, encoding="utf-8"))
    assert q["base"] == "A" and list(q["paired_vs_base"]) == ["B"]
    qq = q["paired_vs_base"]["B"]["quantities"]
    doc = {"schema": "r0d_id_reference/1",
           "arm": "R0d",
           "prereg": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md section 5.1 (FROZEN 2026-10-18)",
           "sources": {"raw/2026-10-18-r0d-outcome/raw/sft.jsonl": {"sha256_lf": sha_lf(sft)}, "raw/2026-10-18-r0d-outcome/raw/rlval_B_vs_ownA_u250.json": {"sha256_lf": sha_lf(u250)}},
           "train_sets": d["train_sets"],
           "epo_con": d["epo_con"],
           "epo": d["epo"],
           "step0_pick": {"base": e["base"]["pick"], "A": e["A"]["pick"], "B": e["B"]["pick"]},
           "u250_contrast_B_minus_ownA_x100": {"TTC": qq["TTC"]["mean"], "PDMS": qq["PDMS"]["mean"]},
           "registered_con_temp_of_R0e_id": 0.1,
           "thresholds": {"step0_pick_points": 0.10, "u250_contrast_points": 0.50}}
    path = os.path.join(pk, "eval", "r0d_id_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"wrote": path, "step0_pick": doc["step0_pick"], "u250": doc["u250_contrast_B_minus_ownA_x100"], "epo_con_keys": sorted(doc["epo_con"])}))


if __name__ == "__main__":
    main()
