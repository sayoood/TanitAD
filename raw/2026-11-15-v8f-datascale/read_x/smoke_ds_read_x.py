"""smoke_ds_read_x.py -- a CODE CHECK of ds_read_x.main on STAND-IN inputs, never a result.  X_s0 = a byte copy of B_rep at BOTH stages (T1 dumps, tick files; the route passes are B_rep's stage-A passes of block 3 and stage-B passes of block 1,
renamed), so that X - B_s0 is exactly B_rep - B_s0 and every wiring error reads wrong.  Two runs: (1) the tick files' VLM goal columns zeroed (the mask held): K-VLM-ZERO must PASS; (2) the ticks unmodified (X scores VLM classes like B_rep):
K-VLM-ZERO must FAIL and the outcome must read CONTROL FAILED.  The checkpoint bank check is stubbed (unit-tested).  Written under readx/smoke/, deleted by the caller (the manifest in the BUILD note lists it).
Expected (run 1): no control problems; K-RE passes at both stages (T1 cells, floors, STRATEGIC, the 0.0767 floor); K1 passes; K-EVALX passes; X - B_s0 TAC-LAT delta equals the banked B_rep - B_s0 row of block 1 / 3 at each stage; the label equals what label_x says
on that real delta and floor (the boundary Delta == -F is reached exactly on real data); X - U_s0 and the four families are present at both stages.
Usage (PYTHONPATH = the read tree's stack and taniteval, DS_BLOCK=4):  python smoke_ds_read_x.py"""
import json
import os
import shutil
import sys

import numpy as np

os.environ["DS_BLOCK"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read_x as X  # noqa: E402

L.bank_problem = lambda *a, **k: None
S = HERE + "/smoke/x"
B1, B3 = L.B1_ROOT, L.B3_ROOT
shutil.rmtree(S, ignore_errors=True)
for d in ("t1/ticks", "t0route", "ev"):
    os.makedirs(f"{S}/{d}")
for st in ("A", "B"):
    shutil.copytree(f"{B1}/t1/dump_{L.t1_tag('B_rep', st)}", f"{S}/t1/dump_{L.t1_tag('X_s0', st)}")
    rd = B3 + "/out/t0route" if st == "A" else B1 + "/out/t0route"
    for v in ("real", "donor", "zero", "realx", "donorx"):
        for ext in ("npz", "json"):
            shutil.copy(f"{rd}/eval_DS-B_rep-{st}_{v}.{ext}", f"{S}/t0route/eval_DS-X_s0-{st}_{v}.{ext}")
    shutil.copy(HERE + "/evidence/X_s0.A.checks.json", f"{S}/ev/X_s0.{st}.checks.json")
k = json.load(open(f"{B1}/out/t0route/KEVALX.json", encoding="utf-8"))
json.dump({**k, "items": {f"X_s0.{st} {v}": k["items"][f"B_rep.B {v}"] for st in ("A", "B") for v in ("real", "donor")}}, open(f"{S}/t0route/KEVALX.json", "w"), indent=1)
GEO = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "STOP_POINT", "SPEED_BAND")
sys.path.insert(0, X.DIAG)
GOAL22 = __import__("diag_read").GOAL22
vlm = [i for i, c in enumerate(GOAL22) if c not in GEO]


def ticks(zero_vlm):
    for st in ("A", "B"):
        z = dict(np.load(f"{B1}/t1/ticks/ticks_DS-B_rep-{st}_m10_causal_heads.npz", allow_pickle=False))
        if zero_vlm:
            p = z["t_goal_prob"].copy()
            p[:, vlm] = 0.0
            z["t_goal_prob"] = p
        np.savez(f"{S}/t1/ticks/ticks_DS-X_s0-{st}_m10_causal_heads.npz", **z)


def run(tag, nb):
    out = f"{S}/{tag}.json"
    rc = X.main(["--dumpsx", S + "/t1", "--t0route", S + "/t0route", "--evidence", S + "/ev", "--out", out, "--n-boot", str(nb)])
    return rc, json.load(open(out))


near = lambda x, y: abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y))  # noqa: E731
ticks(True)
rc1, r1 = run("run1", int(os.environ.get("DS_SMOKE_NB", "2000")))          # B 2,000: the banked block-1 / 3 rows are B 2,000, so K-RE compares every field exactly
c = r1["controls"]
b1 = json.load(open(f"{B1}/out/RESULT_DSCALE10K_B1.json"))
b3 = json.load(open(f"{B3}/out/RESULT_DSCALE10K_B3.json"))
ok = {"rc 0": rc1 == 0, "no problems": r1["controls"]["problems"] == [], "K-RE pass": c["K-RE"]["pass"], "K-RE checks": sorted(c["K-RE"]["per_check"]),
      "K1": c["K1_self_pair"]["synthetic"]["pass"] and c["K1_self_pair"]["X-X.A"]["pass"] and c["K1_self_pair"]["X-X.B"]["pass"], "K-VLM-ZERO pass": c["K-VLM-ZERO"]["pass"],
      "K-RE goal F1": c["K-RE_goal_F1"]["pass"], "C-JOIN-EQ": c["C-JOIN-EQ"]["pass"], "STRATEGIC read": r1["families"]["STRATEGIC"]["block"]["status"] == "OK"}
rows = {st: r1["pairs"][f"X-B_s0.{st}"]["TAC-LAT"] for st in ("A", "B")}
ref = {"A": b3["pairs"]["B_rep-B_s0.A"]["TAC-LAT"] if "B_rep-B_s0.A" in b3["pairs"] else b1["pairs"]["B_rep-B_s0.A"]["TAC-LAT"], "B": b1["pairs"]["B_rep-B_s0.B"]["TAC-LAT"]}
ok["X-B_s0 TAC-LAT delta == banked B_rep - B_s0 (both stages)"] = all(near(rows[s]["delta"], ref[s]["delta"]) and rows[s]["separated"] == ref[s]["separated"] for s in ("A", "B"))
ok["floors == banked"] = near(r1["floors"]["TAC-LAT"]["A"], b1["floors"]["stage_A"]["TAC-LAT"]) and near(r1["floors"]["TAC-LAT"]["B"], b1["floors"]["stage_B"]["TAC-LAT"]) and near(r1["floors"]["TAC-LAT"]["A"], 0.07669862442684461)
d = r1["deciding"]
ok["label from the real delta"] = d["label_stage_A"] == X.label_x(rows["A"]["delta"], rows["A"]["separated"], r1["floors"]["TAC-LAT"]["A"]) and d["label_stage_B"] == X.label_x(rows["B"]["delta"], rows["B"]["separated"], r1["floors"]["TAC-LAT"]["B"])
ok["boundary Delta == -F reached on real data (stage A)"] = (rows["A"]["delta"] == -r1["floors"]["TAC-LAT"]["A"]) and d["label_stage_A"] == ("X-CONTENT" if rows["A"]["separated"] else "X-NO-COST")
ok["four families x 2 pairs x 2 stages"] = all(len(r1["families"][f]["status"]) == 4 for f in X.FAMILIES) and set(r1["families"]["TACTICAL"]["cells"]) == {f"{p}.{s}" for p in X.PAIRS for s in "AB"}
kz1 = c["K-VLM-ZERO"]
ok["K-VLM-ZERO numbers: X 0.0 (VLM zeroed), B_s0 0.12574 and the block-1 floor 0.02737 at stage A"] = near(kz1["x_vlm"], 0.0) and near(kz1["b_s0_vlm"], 0.12574117520688335) and near(kz1["floor"], 0.027371590385418892)
ok["the printed deltas are X - B_s0's, per stage"] = all(near(d["delta"][s_], rows[s_]["delta"]) for s_ in ("A", "B"))
ok["X-U present"] = all("TAC-LAT" in r1["pairs"][f"X-U_s0.{s}"] for s in "AB")
ok["outcome is a label"] = len(r1["outcome"]) == 1 and r1["outcome"][0].startswith(("X-CONTENT", "X-NO-COST", "X-GAIN"))
ticks(False)
rc2, r2 = run("run2", 200)
kz = r2["controls"]["K-VLM-ZERO"]
ok["run 2: K-VLM-ZERO FAILS on unmasked ticks, read marked CONTROL FAILED"] = (not kz["pass"]) and "CONTROL FAILED" in r2["outcome"][0] and rc2 == 1
print(json.dumps({k_: v for k_, v in ok.items()}, indent=1, default=str))
print("run1 outcome:", r1["outcome"], "| deciding:", {k_: d[k_] for k_ in ("label_stage_A", "label_stage_B", "stage_sensitive", "F_cell", "delta")})
print("run2 outcome:", r2["outcome"], "| K-VLM-ZERO:", {k_: kz.get(k_) for k_ in ("x_vlm", "b_s0_vlm", "floor", "pass")})
good = all(v for k_, v in ok.items() if k_ != "K-RE checks")
print("SMOKE_CHECKS", json.dumps({k_: bool(v) for k_, v in ok.items() if k_ != "K-RE checks"}, sort_keys=True))
print("SMOKE OK" if good else "SMOKE FAILED")
sys.exit(0 if good else 1)
