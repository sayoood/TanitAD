#!/usr/bin/env python3
"""rehearse_cs.py -- lead's rehearsal of the reader of card ADD113 on the pod, on BANKED runs only. In each of the three new settings the name of the new run (GEc) is pointed at
the banked reference run RR of that setting, so every engine call of the controls stage (grid read at the pinned grid with no pre-roll and at the grid of the pre-roll of 60 steps,
strategic grid, rule logs with one log and with two) runs once on real files, and the identity control d3 runs on the real banked K30 runs. No rollout of card ADD113 is read and no
value of it is opened. Expected: d1, d3, d5, d6 pass; d0 fails (a reference run has no gap-file meta in its rule log), d4 fails (the lead's test output of the reader kit does not
exist yet), d2 passes or fails on the stand-in lines. Prints the pass flags and the failure texts of the controls only; then runs the names routine on the stand-in part and prints
only the number of name slots. The stand-in names say nothing about the card and are not kept.
Usage (pod, in the working directory and environment of podread_candset10k.sh): rehearse_cs.py <kit dir> <out json>"""
import importlib.util
import json
import os
import sys
import time

V = "/workspace/v8proof"
W, P, GE, CS = V + "/integ10k", V + "/podread", V + "/candset", V + "/cropscale"
RD, CODE = P + "/root/integ10k", P + "/code"
CHK = V + "/visrule10k/check_v"
K, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("read_candset10k", K + "/read_candset10k.py")
RF = importlib.util.module_from_spec(spec)
sys.modules["read_candset10k"] = RF
spec.loader.exec_module(RF)
for S in RF.NEW_SETTINGS:
    RF.SPEC[S]["runs"]["GEc"] = RF.SPEC[S]["runs"]["RR"]
LT = GE + "/lead_tests"
now = str(int(time.time()))
rc = RF.main(["--stage", "controls", "--model-reader", CS + "/kit_read/read_cropscale10k.py", "--pull", RD + "/pull", "--code", CODE, "--grid-dir", W + "/longroll",
              "--grid-pinned", V + "/hierrep/pins/eval_grid_windows.json", "--w", W, "--tree", W + "/tree", "--wrapper-file", W + "/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py",
              "--gap-file", CS + "/gapfiles/VGAPLE.npz", "--gap-record", V + "/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json", "--wrapper-check", CHK + "/CHECK_WRAPPER_V_DONE.txt",
              "--wrapper-hashes", CHK + "/CHECK_WRAPPER_V.hashes", "--tests-queue", LT + "/lead_pytest_queue.out", "--tests-read", LT + "/lead_pytest_read.out",
              "--tests-mtime-queue", str(int(os.path.getmtime(LT + "/lead_pytest_queue.out"))), "--tests-mtime-read", "0",
              "--queue-start", open(GE + "/CANDSET10K_START.txt").readline().split()[0], "--controls-start", now, "--done-marker", GE + "/CANDSET10K_START.txt", "--out", out])
print("REHEARSAL rc", rc)
d = json.load(open(out, encoding="utf-8"))
for k in sorted(d.get("controls", {})):
    c = d["controls"][k]
    print("REHEARSAL", k, "pass", c.get("pass"), "| failures:", " ;; ".join(str(x)[:220] for x in (c.get("failures") or [])[:4]))
part = d.get("payload") or d.get("withheld") or {}
print("REHEARSAL keys of the value part:", sorted(part.keys()))
try:
    names = RF.all_names(part)
    flat = RF.flat_names(names)
    json.dumps(names)
    print("REHEARSAL names routine on the stand-in part ran through, name slots", len(flat))
except Exception as e:  # the rehearsal reports, it does not repair
    print("REHEARSAL names routine FAILED:", type(e).__name__, str(e)[:300])
