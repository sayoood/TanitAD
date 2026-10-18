#!/usr/bin/env python3
"""Run ONE spec on ONE tree exactly as battery.py's run() does (the run() source is exec'd from battery.py itself, only OUT/TREES rebound).
Usage: run1.py <tree:fix|base|<abs tree path>> <spec> <log_dir> <tag> [timeout_s]"""
import json, os, re, subprocess, sys, time
SRC = open("C:/Users/Admin/p6gate/runner/battery.py", encoding="utf-8").read()
head = SRC.split("os.makedirs(OUT", 1)[0]                     # constants + imports
fns = re.search(r"(def name_of\(spec\):.*?)\ndef lane\(", SRC, re.S).group(1)
tree, spec, outd, tag = sys.argv[1:5]
tmo = int(sys.argv[5]) if len(sys.argv) > 5 else 300
g = {"__name__": "battery_src"}
sys.argv = ["battery.py", outd]
exec(compile(head, "battery.py", "exec"), g)
g["OUT"] = outd
os.makedirs(outd, exist_ok=True)
if tree not in g["TREES"]:
    g["TREES"][tree] = tree
fns = fns.replace("timeout=3600", f"timeout={tmo}").replace('rc = "TIMEOUT_3600s"', f'rc = "TIMEOUT_{tmo}s"')
fns = fns.replace('log = os.path.join(OUT, f"{tree}__{nm}.log")', 'log = os.path.join(OUT, f"{TAG}__{nm}.log")')
g["TAG"] = tag
g["res"] = {tree: {}}
import threading; g["lock"] = threading.Lock(); g["SUMM"] = os.path.join(outd, f"{tag}_summary.json")
exec(compile(fns, "battery.py", "exec"), g)
g["run"](tree, spec)
r = g["res"][tree]
print(tag, json.dumps(r))
