"""floor_table.py -- every row of out/READ_FULL.json against the replicate floor (rule r4: |OB-BAND_rep - OB-BAND_s0|, read at seed 0,
applied to both seeds).  Band effect (BAND - CTL) and the three ladder steps (beside, post-hoc).  Writes out/FLOOR_TABLE.json; derived
from READ_FULL.json only, no new estimate.  Usage: python floor_table.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out/READ_FULL.json"), encoding="utf-8"))
OP, TAC = "operative_0_2s", "tactical_2_6s"
rep = d["replicate"]["rows_REP_minus_BAND_s0"]
out = {"source": "out/READ_FULL.json", "floor_rule": d["replicate"]["floor_rule"],
       "note": "a row 'clears' when it is separated AND |delta| exceeds the floor; the floor is one replicate pair (seed 2 vs seed 0, init M2b s0): "
               "a run-to-run estimate of the band recipe, applied here to the ladder's DAG-recipe steps too, which is an assumption stated, not measured",
       "rows": {}}
for band in (OP, TAC):
    for m, r in rep[band].items():
        if r.get("status") != "OK":
            continue
        fl = abs(r["delta"])
        row = {"floor": fl, "rep_separated": r["separated"]}
        for s in ("s0", "s1"):
            R = d["seeds"][s]["bands"][band]["rows"][m]
            e = R["BAND_minus_CTL"]
            row[f"BAND-CTL {s}"] = {"delta": e["delta"], "separated": e["separated"], "clears": bool(e["separated"] and abs(e["delta"]) > fl)}
            L = R.get("ladder_post_hoc")
            if L:
                for k, nm in (("step1_schedule_DAG0_minus_INIT", "step1 schedule"), ("step2_dagger_rows_DAG_minus_DAG0", "step2 DAgger rows"),
                              ("step3_w_acc_DAGOACC_minus_DAG", "step3 w_acc")):
                    x = L[k]
                    row[f"{nm} {s}"] = {"delta": x["delta"], "separated": x["separated"], "clears": bool(x["separated"] and abs(x["delta"]) > fl)}
        out["rows"][f"{band}:{m}"] = row
json.dump(out, open(os.path.join(HERE, "out/FLOOR_TABLE.json"), "w", encoding="utf-8"), indent=1)
keys = [k for k in next(iter(out["rows"].values())) if k not in ("floor", "rep_separated")]
print("row".ljust(34), "floor   ", " | ".join(k for k in keys))
for nm, row in out["rows"].items():
    cells = []
    for k in keys:
        c = row[k]
        cells.append(f"{c['delta']:+.3f}{'#' if c['clears'] else ('*' if c['separated'] else ' ')}")
    print(nm.ljust(34), f"{row['floor']:.4f}", " | ".join(cells))
print("# separated and above the floor; * separated, within the floor")
