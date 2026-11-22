#!/usr/bin/env python3
"""rehearse_pa.py -- lead's rehearsal of the controls stage of the readers of cards ADD114 and ADD115 on the pod, on BANKED runs only. The name of each new run is pointed at a
banked run of the same setting (a stand-in), so every engine call of the controls stage (grid read, strategic grid, rule logs) runs once on real files and the identity control
runs on the real banked runs. No rollout of card ADD114 or ADD115 is read and no value of them is opened. Expected: the wrapper, identity and engine controls pass; the rule-log
control fails on the stand-ins as it has to (a stand-in for LA has the gap-file meta of another file, a stand-in for PL received the label gap and not the planted one); the
counts control fails on the done-marker text (the start marker is passed in its place); the arguments control passes or fails on the stand-in lines.
Prints the pass flags and the failure texts of the controls only; then runs the names routine on the stand-in part and prints only the number of name slots.
Usage (pod, in the working directory and environment of the pod driver): rehearse_pa.py <reader file> <root of the card on the pod> <marker file with the start epoch> <out json> <stand-ins json>
  stand-ins json: {"<setting>": {"<arm>": "<banked run name without I10>"}}"""
import importlib.util
import json
import os
import sys
import time

V = "/workspace/v8proof"
W, P, CS = V + "/integ10k", V + "/podread", V + "/cropscale"
RD, CODE = P + "/root/integ10k", P + "/code"
CHK = V + "/visrule10k/check_v"
reader, GE, marker, out, stand = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("reader_under_rehearsal", reader)
RF = importlib.util.module_from_spec(spec)
sys.modules["reader_under_rehearsal"] = RF
spec.loader.exec_module(RF)
for S, arms in json.loads(stand).items():
    for arm, run in arms.items():
        assert arm in RF.NEW_ARMS[S], (S, arm)
        RF.SPEC[S]["runs"][arm] = run
LT = GE + "/lead_tests"
now = str(int(time.time()))
rc = RF.main(["--stage", "controls", "--model-reader", CS + "/kit_read/read_cropscale10k.py", "--pull", RD + "/pull", "--code", CODE, "--grid-dir", W + "/longroll",
              "--grid-pinned", V + "/hierrep/pins/eval_grid_windows.json", "--w", W, "--tree", W + "/tree", "--wrapper-file", W + "/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py",
              "--gap-file", CS + "/gapfiles/VGAPLE.npz", "--gap-record", V + "/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json", "--la-file", V + "/fallattr/gapfiles/VGAPA_LA.npz",
              "--jle-file", V + "/presattr/joins/JLE.npz", "--jle-record", V + "/presattr/joins/JLE_RECORD.json", "--eval-join", V + "/hierrep/pins/join_eval141.npz",
              "--wrapper-check", CHK + "/CHECK_WRAPPER_V_DONE.txt", "--wrapper-hashes", CHK + "/CHECK_WRAPPER_V.hashes",
              "--tests-queue", LT + "/lead_pytest_queue.out", "--tests-read", LT + "/lead_pytest_read.out",
              "--tests-mtime-queue", str(int(os.path.getmtime(LT + "/lead_pytest_queue.out"))), "--tests-mtime-read", str(int(os.path.getmtime(LT + "/lead_pytest_read.out"))),
              "--queue-start", open(marker).readline().split()[0], "--controls-start", now, "--done-marker", marker, "--out", out])
print("REHEARSAL rc", rc)
d = json.load(open(out, encoding="utf-8"))
print("REHEARSAL card", d.get("card"), "all_pass", d.get("all_pass"))
for k in sorted(d.get("controls", {})):
    c = d["controls"][k]
    print("REHEARSAL", k, "pass", c.get("pass"), "| failures:", " ;; ".join(str(x)[:200] for x in (c.get("failures") or [])[:3]))
part = d.get("payload") or d.get("withheld") or {}
print("REHEARSAL keys of the value part:", sorted(part.keys()), "settings", sorted((part.get("settings") or {}).keys()))
try:
    names = RF.all_names(part)
    flat = RF.flat_names(names)
    json.dumps(names)
    print("REHEARSAL names routine on the stand-in part ran through, name slots", len(flat))
except BaseException as e:  # the rehearsal reports, it does not repair
    print("REHEARSAL names routine FAILED:", type(e).__name__, str(e)[:300])
