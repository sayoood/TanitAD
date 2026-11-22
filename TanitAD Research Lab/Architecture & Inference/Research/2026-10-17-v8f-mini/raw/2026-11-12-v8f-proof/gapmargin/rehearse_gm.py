#!/usr/bin/env python3
"""rehearse_gm.py -- lead's rehearsal of the controls stage of the reader of card ADD117 on the pod, on BANKED runs only, after the queue has made the scaled gap files
and before its rollouts are read. The name of each new run is pointed at the banked candidate run GEc of its setting (a stand-in), so every engine call of the controls
stage runs once on real files, the identity control runs on the real banked runs, and the first control reads the real record of the scaled files. No rollout of card
ADD117 is read and no value of it is opened. Expected: the first control (gap files), the wrapper, the arguments (a stand-in is its own reference), identity, tests and
engine controls pass; the rule-log control fails on the stand-ins as it has to (their logs name the base file, not a scaled one); the counts control fails on the
done-marker text (the start marker is passed in its place).
Prints the pass flags and the failure texts of the controls only; then runs the names routine on the stand-in part and prints only the number of name slots.
Usage (pod, in the working directory and environment of the pod driver): rehearse_gm.py <reader file> <out json>"""
import importlib.util
import json
import os
import sys
import time

V = "/workspace/v8proof"
W, P, CS, GE = V + "/integ10k", V + "/podread", V + "/cropscale", V + "/gapmargin"
RD, CODE = P + "/root/integ10k", P + "/code"
CHK = V + "/visrule10k/check_v"
reader, out = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("reader_under_rehearsal", reader)
RF = importlib.util.module_from_spec(spec)
sys.modules["reader_under_rehearsal"] = RF
spec.loader.exec_module(RF)
for S in RF.SETTINGS:
    for arm in RF.NEW_ARMS[S]:
        RF.SPEC[S]["runs"][arm] = RF.SPEC[S]["runs"]["GEc"]
LT = GE + "/lead_tests"
marker = GE + "/GAPMARGIN10K_START.txt"
now = str(int(time.time()))
rc = RF.main(["--stage", "controls", "--model-reader", CS + "/kit_read/read_cropscale10k.py", "--pull", RD + "/pull", "--code", CODE, "--grid-dir", W + "/longroll",
              "--grid-pinned", V + "/hierrep/pins/eval_grid_windows.json", "--w", W, "--tree", W + "/tree", "--wrapper-file", W + "/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py",
              "--gap-file", CS + "/gapfiles/VGAPLE.npz", "--margin-dir", GE + "/gapfiles", "--margin-record", GE + "/gapfiles/GAPMARGIN_GAPFILES_RECORD.json",
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
    lines = RF.print_sections(dict(part, control_values=part.get("control_values") or {}), names)
    print("REHEARSAL names routine and print routine on the stand-in part ran through, name slots", len(flat), "print lines", len(lines))
except BaseException as e:  # the rehearsal reports, it does not repair
    print("REHEARSAL names routine FAILED:", type(e).__name__, str(e)[:300])
