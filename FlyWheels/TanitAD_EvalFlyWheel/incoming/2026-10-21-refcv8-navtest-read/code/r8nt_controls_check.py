#!/usr/bin/env python3
"""r8nt_controls_check.py -- the KNOWN-VALUE controls of the refcv8 navtest read, scored on the Thor backend (navsim-cpu python, pandas).

    python r8nt_controls_check.py <scores_dir> <tag> <W3 STOP_navtest.csv> <banked Thor r7thor_s50400_R7_VMAXOFF.csv> <out.json>

Two controls; BOTH must PASS before the full roll is launched (brief: "if a control misses, stop and report -- do not launch"):
  * STOP  (official zero plan, model-free): full-split PDMS x100 (4 dp) must read 61.8202 (W3's banked dev-box value, INHERITED) AND
    per token, every term column must equal W3's banked CSV within 1e-9 (the backend policy's cross-backend tolerance) with 0 flips.
  * R7    (refcv7-50,400 LEGAL = R7_VMAXOFF seam sha256 5b65d31f..., re-scored): PDMS x100 must read 70.2048 (MODEL_REGISTRY 15.4.1,
    INHERITED) AND per token equal the banked Thor CSV (sha256 166ca2a0...) within 1e-9.
Literal expected values, never derived from the run under test.
"""
import json
import sys

import pandas as pd

TERMS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
         "driving_direction_compliance", "score"]
EXPECT = {"STOP": 61.8202, "R7": 70.2048}
TOL = 1e-9


def load(p):
    df = pd.read_csv(p, index_col=0, float_precision="round_trip")
    df = df[df["token"] != "average"].set_index("token")
    return df


def compare(got, ref):
    rec = {"n_got": int(len(got)), "n_ref": int(len(ref)), "same_token_set": set(got.index) == set(ref.index)}
    if not rec["same_token_set"]:
        rec["pass"] = False
        return rec
    ref = ref.loc[got.index]
    mx, flips = {}, {}
    for t in TERMS:
        a, b = got[t].astype(float), ref[t].astype(float)
        mx[t] = float((a - b).abs().max())
        flips[t] = int(((a > 0.5) != (b > 0.5)).sum()) if t not in ("ego_progress", "score") else None
    rec["max_abs_diff"] = mx
    rec["discrete_flips"] = flips
    rec["pdms_x100_4dp"] = round(100.0 * float(got["score"].mean()), 4)
    rec["pass_per_token"] = all(v <= TOL for v in mx.values()) and all((f or 0) == 0 for f in flips.values())
    return rec


scores, tag, stop_ref, r7_ref, out = sys.argv[1:6]
res = {"what": "refcv8 navtest read -- known-value controls on the Thor CPU backend (VALIDATION, never a refcv8 number)",
       "tolerance_per_token": TOL, "expected_pdms_x100": EXPECT, "controls": {}}
for name, ref in (("STOP", stop_ref), ("R7", r7_ref)):
    lab = f"r8nt_{tag}_{name}"
    try:
        c = json.load(open(f"{scores}/{lab}/{lab}.counts.json", encoding="utf-8"))
        got = load(f"{scores}/{lab}/{lab}.csv")
        rec = compare(got, load(ref))
        rec["counts_status"] = c.get("status")
        rec["counts_pdms_x100_4dp"] = (c.get("summary_x100_4dp") or {}).get("PDMS")
        rec["csv_sha256"] = c.get("csv_sha256")
        rec["pass"] = bool(rec.get("pass_per_token")) and c.get("status") == "PASS" and \
            rec["counts_pdms_x100_4dp"] == EXPECT[name] and rec["pdms_x100_4dp"] == EXPECT[name]
    except Exception as e:  # noqa: BLE001
        rec = {"pass": False, "error": repr(e)[:400]}
    res["controls"][name] = rec
res["verdict"] = "PASS" if all(r.get("pass") for r in res["controls"].values()) else "FAIL"
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print("QQCTRL " + json.dumps({"verdict": res["verdict"], **{k: (v.get("pass"), v.get("pdms_x100_4dp")) for k, v in res["controls"].items()}}))
