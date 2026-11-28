"""smoke_ds_read_b2.py -- a CODE CHECK of ds_read_b2.main on STAND-IN inputs, never a result: seed-1 dumps are byte copies of the seed-0 dumps of the same role (B_s1 = copy of B_s0, U_s1 = copy of U_s0) under the seed-1 names,
the T0 passes are absent, and the block-1 result is produced by running the block-1 reader (ds_read.py, CPU, B 200) on the real block-1 dumps (read-only).  Expected: K-RE (B1) passes on both pairs, the seed-1 rows equal the
seed-0 rows exactly (same dumps), the cross-venue seed-1 beside is present, the label is NOT-YET (the T0 evidence is absent).  Everything is written under read2/smoke/ and deleted by the caller.
Usage (PYTHONPATH = the read tree's stack and taniteval):  python smoke_ds_read_b2.py"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R1  # noqa: E402
import ds_read_b2 as R2  # noqa: E402

S = HERE + "/smoke"
shutil.rmtree(S, ignore_errors=True)
os.makedirs(S + "/t1")
for arm0, arm1 in (("B_s0", "B_s1"), ("U_s0", "U_s1")):
    for st in ("A", "B"):
        shutil.copytree(f"{L.B1_ROOT}/t1/dump_{L.t1_tag(arm0, st)}", f"{S}/t1/dump_{L.t1_tag(arm1, st)}")
none = S + "/none"
rc1 = R1.main(["--dumps", L.B1_ROOT + "/t1", "--t0mech", none, "--t0route", none, "--out", S + "/B1.json", "--n-boot", "200"])
rc2 = R2.main(["--dumps0", L.B1_ROOT + "/t1", "--dumps1", S + "/t1", "--t0mech0", none, "--t0route0", none, "--t0mech1", none, "--t0route1", none, "--b1-result", S + "/B1.json", "--out", S + "/B2.json", "--n-boot", "200"])
r = json.load(open(S + "/B2.json"))
k = r["controls"].get("K-RE_B1", {})
same = all(r["pairs"][f"U-B_s1.{st}"] == r["pairs"][f"U-B_s0.{st}"] for st in ("A", "B"))
print("b1 rc", rc1, "| b2 rc", rc2, "| outcome", r["outcome"], "| K-RE_B1", k.get("pass"), {p: v["rows_compared"] for p, v in k.get("per_pair", {}).items()},
      "| seed-1 rows == seed-0 rows:", same, "| cross-venue seed 1:", sorted(r["cross_venue_beside"]["seed_1"])[:3], "| problems:", r["controls"]["problems"])
ok = bool(k.get("pass")) and same and "abs_diff" in r["cross_venue_beside"]["seed_1"] and r["outcome"][0].startswith(("NOT-YET", "BLOCKED"))
print("SMOKE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
