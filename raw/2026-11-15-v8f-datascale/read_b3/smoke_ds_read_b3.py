"""smoke_ds_read_b3.py -- a CODE CHECK of ds_read_b3.main on STAND-IN inputs, never a result.  UE_s0 = a byte copy of R_s0 and BL_s0 = a byte copy of B_rep (T1 dumps and stage-A mechanism evals; the five route passes are the block-1 stage-B passes of
U_s0 and B_rep renamed to stage A; R_s0 has none), so that every pair is a different pair of real arms and a mis-wired pair or reference reads wrong, under the block-3 names.  The block-1 result is produced by running the block-1 reader (ds_read.py, CPU, B 200) on the real block-1 dumps and T0 evals
(read-only), and stands in for the block-2 result as well, so the block-1 stage-A rows reused under K-RE come from the same bootstrap size.  The checkpoint bank check is stubbed (unit-tested).
Expected: no control problems; K-RE passes (T1 cells at B1 and B2, the ADE6 means, the mechanism rows); STRATEGIC read (status OK); BL - B_s0 equals B_rep - B_s0 and UE - U_s0 equals the block-1 R_s0 - U_s0 row cell by cell;
the label's deltas equal the differences of the independently read ADE6 means and F_A equals |mean(B_rep) - mean(B_s0)|; the label is a card label or READING-SENSITIVE (the readings are different pairs now).  Written under read3/smoke/, deleted by the caller.
Usage (PYTHONPATH = the read tree's stack and taniteval, DS_BLOCK=3):  python smoke_ds_read_b3.py"""
import json
import os
import re
import shutil
import sys

os.environ["DS_BLOCK"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R1  # noqa: E402
import ds_read_b3 as R3  # noqa: E402
import sync_b1 as SY  # noqa: E402

L.bank_problem = lambda *a, **k: None
S = HERE + "/smoke/b3"
B1 = L.B1_ROOT
shutil.rmtree(S, ignore_errors=True)
for d in ("t1", "t0mech", "t0route"):
    os.makedirs(f"{S}/{d}")
for new, old in (("UE_s0", "R_s0"), ("BL_s0", "B_rep")):
    shutil.copytree(f"{B1}/t1/dump_{L.t1_tag(old, 'A')}", f"{S}/t1/dump_{L.t1_tag(new, 'A')}")
c, s, bad = SY.sync(B1 + "/out/t0mech", S + "/t0mech")
assert not bad, bad
for new, old in (("UE_s0", "R_s0"), ("BL_s0", "B_rep")):
    n0, n1 = "DS" + old.replace("_", "") + "A", "DS" + new.replace("_", "") + "A"
    for f in os.listdir(f"{S}/t0mech"):
        if f.startswith(f"eval_{n0}_s0"):
            shutil.copy(f"{S}/t0mech/{f}", f"{S}/t0mech/{f.replace(n0, n1, 1)}")
src = {"B_s0": "B_s0", "U_s0": "U_s0", "B_rep": "B_rep", "UE_s0": "U_s0", "BL_s0": "B_rep"}      # route passes (stage-B files of the named block-1 arm, renamed)
for arm, old in src.items():
    for v in ("real", "donor", "zero", "realx", "donorx"):
        for ext in ("npz", "json"):
            shutil.copy(f"{B1}/out/t0route/eval_DS-{old}-B_{v}.{ext}", f"{S}/t0route/eval_DS-{arm}-A_{v}.{ext}")
k = json.load(open(f"{B1}/out/t0route/KEVALX.json", encoding="utf-8"))
json.dump({**k, "items": {f"{a}.A {v}": k["items"][f"{o}.B {v}"] for a, o in src.items() for v in ("real", "donor")}}, open(f"{S}/t0route/KEVALX.json", "w"), indent=1)
rc1 = R1.main(["--dumps", B1 + "/t1", "--t0mech", B1 + "/out/t0mech", "--t0route", B1 + "/out/t0route", "--out", S + "/B1.json", "--n-boot", "200"])
rc3 = R3.main(["--dumps3", S + "/t1", "--t0mech", S + "/t0mech", "--t0route", S + "/t0route", "--b1-result", S + "/B1.json", "--b2-result", S + "/B1.json", "--out", S + "/B3.json", "--n-boot", "200"])
r = json.load(open(S + "/B3.json"))
c = r["controls"]
cells = lambda p: {k_: v for k_, v in r["pairs"][p].items() if v.get("status") == "OK"}  # noqa: E731
near = lambda x, y: abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y))  # noqa: E731
b1r = json.load(open(S + "/B1.json"))
t1 = [c_ for c_ in cells("BL-B_s0.A") if c_ in r["pairs"]["B_rep-B_s0.A"] and r["pairs"]["B_rep-B_s0.A"][c_].get("status") == "OK" and c_.split(":")[0] not in ("S1", "S2")]
bl_ok = bool(t1) and all(near(r["pairs"]["BL-B_s0.A"][c_]["delta"], r["pairs"]["B_rep-B_s0.A"][c_]["delta"]) for c_ in t1)
ue_cells = [c_ for c_ in b1r["pairs"]["R-U_s0.A"] if b1r["pairs"]["R-U_s0.A"][c_].get("status") == "OK"]
ue_ok = bool(ue_cells) and all(near(r["pairs"]["UE-U_s0.A"][c_]["delta"], b1r["pairs"]["R-U_s0.A"][c_]["delta"]) for c_ in ue_cells)
ade = r["ade6_stage_A_T0"]
mn, lab = ade["mean_ade6_m"], r["h_data_scale_1"]
ade_ok = (near(ade["F_A"], abs(mn["B_rep"] - mn["B_s0"])) and near(lab["delta_ade6_U_minus_BL"], mn["U_s0"] - mn["BL_s0"]) and near(lab["delta_ade6_UE_minus_BL"], mn["UE_s0"] - mn["BL_s0"])
          and near(lab["delta_data_U_minus_B"], mn["U_s0"] - mn["B_s0"]) and near(lab["delta_ep_BL_minus_B"], mn["BL_s0"] - mn["B_s0"]) and lab["F_A"] == ade["F_A"])
strat = r["families"]["STRATEGIC"]["block"]
kre = c.get("K-RE", {})
same, zero_ue, zero_bl, ade_same = ue_ok, ue_ok, bl_ok, ade_ok
print("b1 rc", rc1, "| b3 rc", rc3, "| outcome", r["outcome"], "| K-RE", kre.get("pass"), {k_: v.get("rows_compared") for k_, v in kre.get("per_check", {}).items()}, "| UE-U == block-1 R-U", ue_ok, "| BL-B == B_rep-B", bl_ok, "| ADE6 label deltas == mean differences", ade_ok, "| strat", strat.get("status"), "| F_A", ade.get("F_A"), "| problems", c["problems"])
ok = (rc1 == 0 and rc3 == 0 and not c["problems"] and bool(kre.get("pass")) and len(kre.get("per_check", {})) >= 4 and zero_ue and zero_bl and same and ade_same and strat.get("status") == "OK"
      and r["outcome"][0] in R3.LABELS + ("READING-SENSITIVE",) and "stage_A_T_floor" in r["mechanism_stage_A_T0"] and len(r["mechanism_stage_A_T0"]["pairs"]) == 6)
print("SMOKE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
