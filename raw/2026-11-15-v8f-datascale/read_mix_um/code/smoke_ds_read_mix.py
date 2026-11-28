"""smoke_ds_read_mix.py -- a CODE CHECK of ds_read_mix.main on STAND-IN inputs with KNOWN ANSWERS, never a result (card section 6, the two stand-in smokes).
  python smoke_ds_read_mix.py bl   UM := BL_s0's dumps, mechanism evals and route passes (byte copies renamed to UM_s0).  Known answer: DILUTION-CONFIRMED; UM - UE TAC-LAT == BL - UE == the mirror of the banked UE - BL row
                                   (+0.0855), UM - BL reads 0.0 [0, 0] on every cell, K1 / K-RE (B3) / K-EVALX pass, no control problems.
  python smoke_ds_read_mix.py ue   UM := UE_s0's.  Known answer: NOT-DILUTION; UM - UE reads 0.0 [0, 0] on every cell, UM - BL equals the banked UE - BL row, same controls.
Run at the full bootstrap size (B 2,000, seed 0) so that K-RE (B3) compares exactly against RESULT_DSCALE10K_B3.json.  The checkpoint bank check is stubbed (unit-tested); the UM summary / gate / replay inputs are the REAL ones.
The reference arms are read from read / read3 (read only); the stand-in folder smoke/<which> is deleted by the caller.  PYTHONPATH = the read tree's stack and taniteval, DS_BLOCK=mix, CUDA_VISIBLE_DEVICES=-1."""
import json
import os
import shutil
import sys

os.environ["DS_BLOCK"] = "mix"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read_mix as M  # noqa: E402
import sync_refs as SY  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "bl"
SRC = {"bl": "BL_s0", "ue": "UE_s0"}[which]
L.bank_problem = lambda *a, **k: None
S = f"{HERE}/smoke/{which}"
B3 = L.B3_ROOT
shutil.rmtree(S, ignore_errors=True)
for d in ("t1/ticks", "t0mech", "t0route"):
    os.makedirs(f"{S}/{d}")
tag_um, tag_src = L.t1_tag("UM_s0", "A"), L.t1_tag(SRC, "A")
shutil.copytree(f"{B3}/t1/dump_{tag_src}", f"{S}/t1/dump_{tag_um}")
shutil.copy(f"{B3}/t1/ticks/ticks_{tag_src}.npz", f"{S}/t1/ticks/ticks_{tag_um}.npz")
c, s, bad = SY.sync(B3 + "/out/t0mech", S + "/t0mech")
assert not bad, bad
n0, n1 = "DS" + SRC.replace("_", "") + "A", "DSUMs0A"
for f in sorted(os.listdir(f"{S}/t0mech")):
    if f.startswith(f"eval_{n0}_s0"):
        shutil.copy(f"{S}/t0mech/{f}", f"{S}/t0mech/{f.replace(n0, n1, 1)}")
for v in ("real", "donor", "zero", "realx", "donorx"):
    for ext in ("npz", "json"):
        shutil.copy(f"{B3}/out/t0route/eval_DS-{SRC}-A_{v}.{ext}", f"{S}/t0route/eval_DS-UM_s0-A_{v}.{ext}")
k = json.load(open(f"{B3}/out/t0route/KEVALX.json", encoding="utf-8"))
json.dump({**k, "items": {f"UM_s0.A {v}": k["items"][f"{SRC}.A {v}"] for v in ("real", "donor")}}, open(f"{S}/t0route/KEVALX.json", "w"), indent=1)
rc = M.main(["--dumpsm", S + "/t1", "--t0mech", S + "/t0mech", "--t0route", S + "/t0route", "--um-summary", M.CARD_DIR + "/um_replay/UM_s0.A.summary.json", "--out", S + "/MIX.json"])
r = json.load(open(S + "/MIX.json"))
c = r["controls"]
b3 = json.load(open(B3 + "/out/RESULT_DSCALE10K_B3.json", encoding="utf-8"))
tl = lambda p: r["pairs"][p]["TAC-LAT"]  # noqa: E731
near = lambda x, y: abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y))  # noqa: E731
ue_bl = b3["pairs"]["UE-BL.A"]["TAC-LAT"]
zero_all = lambda p: all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in r["pairs"][p].values() if v.get("status") == "OK")  # noqa: E731
if which == "bl":
    known = (r["outcome"][0].startswith("DILUTION-CONFIRMED") and near(tl("UM-UE.A")["delta"], -ue_bl["delta"]) and tl("UM-UE.A")["delta"] > 0.0767 and zero_all("UM-BL.A")
             and near(tl("UM-UE.A")["lo"], -ue_bl["hi"]) and near(tl("UM-UE.A")["hi"], -ue_bl["lo"]) and tl("UM-BL.A")["hi"] == 0.0 and tl("UM-BL.A")["lo"] == 0.0)
else:
    known = (r["outcome"][0].startswith("NOT-DILUTION") and zero_all("UM-UE.A") and near(tl("UM-BL.A")["delta"], ue_bl["delta"]) and near(tl("UM-BL.A")["lo"], ue_bl["lo"]) and near(tl("UM-BL.A")["hi"], ue_bl["hi"]))
tagged = all(M.A1_TAG in x and "10 > 5, BLOCKED as committed" in x for x in r["outcome"]) and M.A1_TAG in open(S + "/MIX.md", encoding="utf-8").read()
kre = c.get("K-RE_B3", {})
print("which", which, "| rc", rc, "| outcome", r["outcome"], "| UM-UE TAC-LAT", tl("UM-UE.A"), "| UM-BL TAC-LAT", tl("UM-BL.A"), "| banked UE-BL TAC-LAT", ue_bl)
print("K-RE", kre.get("pass"), {k_: v.get("rows_compared") for k_, v in kre.get("per_check", {}).items()}, "| K1", c["K1_self_pair"]["pass"], "| K-EVALX", c["K-EVALX"], "| problems", c["problems"], "| blocked", r["blocked_items"],
      "| K-TACSRC under UM-A1", r["k_tacsrc"]["pass_under_UM_A1"], "| strat", r["families"]["STRATEGIC"]["block"]["status"], "| f1", r["goal_f1_beside"].get("status"), r["goal_f1_beside"].get("reason"))
print("known answer:", known, "| UM-A1 line beside every label:", tagged, "| beside", r["beside_labels"])
ok = (known and tagged and not c["problems"] and not r["blocked_items"] and bool(kre.get("pass")) and len(kre.get("per_check", {})) >= 9 and c["K1_self_pair"]["pass"] and r["k_tacsrc"]["pass_under_UM_A1"]
      and r["families"]["STRATEGIC"]["block"]["status"] == "OK" and "stage_A_T_floor" in r["mechanism_stage_A_T0"] and len(r["mechanism_stage_A_T0"]["pairs"]) == len(M.PAIRS) and rc == 0)
print("SMOKE", which, "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
