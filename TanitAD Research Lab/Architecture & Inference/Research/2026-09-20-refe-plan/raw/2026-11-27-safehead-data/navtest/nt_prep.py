#!/usr/bin/env python3
"""NAVTEST dump prep: job + commands + anchor picks for ALL navtest tokens (the anchor table's 12,146 rows). No labels, no PDMS.
Writes F:/safehead_cache/navtest/plan/{job_navtest.json,commands_navtest.json,anchor_pick.json}. Logs sorted; tokens sorted within a log."""
import glob, json, os, sys
sys.path.insert(0, "F:/safehead_cache/code")
import sh_lib as L
ANCH = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
FR = "E:/Projects/TanitAD/data/refe_navtest/frames"
PL = L.wp("navtest", "plan")
os.makedirs(PL, exist_ok=True)
pick, logof = {}, {}
for f in sorted(glob.glob(ANCH)):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        pick[r["token"]] = int(r["pick"]); logof[r["token"]] = r["log"]
exp = L.load_json_gz(EXPORT)["tokens"]
miss = [t for t in pick if t not in exp]
assert len(pick) == 12146 and not miss, (len(pick), len(miss))
by = {}
for t, lg in logof.items():
    assert exp[t]["log_name"] == lg
    by.setdefault(lg, []).append(t)
logs, cmds = [], {}
for lg in sorted(by):
    db = f"{L.DB_ROOT}/test/{lg}.db"
    assert os.path.exists(db), db
    assert all(os.path.exists(f"{FR}/{lg}_{c}.zip") for c in L.CAMS), lg
    toks = sorted(by[lg])
    logs.append({"log": lg, "db": db, "split": "navtest", "tokens": toks})
    for t in toks:
        cmds[f"{lg}/{t}"] = exp[t]["ego_statuses"][-1]["driving_command"]
json.dump({"frames": FR, "yaw": L.YAW_TEST, "commands": f"{PL}/commands_navtest.json", "logs": logs}, open(f"{PL}/job_navtest.json", "w"))
json.dump(cmds, open(f"{PL}/commands_navtest.json", "w"))
json.dump(pick, open(f"{PL}/anchor_pick.json", "w"))
print("prep ok", len(logs), "logs", len(pick), "tokens")
