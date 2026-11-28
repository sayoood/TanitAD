"""AGENT J finisher (no model turns): waits for the full suite and the mutation run, then writes PINS_J.txt and logs/J_FINISHED.txt (the two verdict lines). Cap 90 min."""
import os
import re
import subprocess
import sys
import time

T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def tail(p, n=600):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


t0 = time.time()
while time.time() - t0 < 5400:
    mut, suite = tail(f"{T}/logs/mutations_refs.log"), tail(f"{T}/logs/pytest_stage2_refs_full.log")
    if re.search(r"^rc=\d+", mut, re.M) and re.search(r"\d+ (passed|failed)", suite):
        break
    time.sleep(20)
mut, suite = tail(f"{T}/logs/mutations_refs.log", 3000), tail(f"{T}/logs/pytest_stage2_refs_full.log", 1500)
red = re.findall(r"mutations RED: \d+ of \d+", mut)
res = re.findall(r"=+ .*? in [\d.]+s.*?=+|\d+ passed[^\n]*|\d+ failed[^\n]*", suite)
subprocess.run([PY, "-I", f"{T}/audit/make_pins_refs.py"], capture_output=True, text=True)
with open(f"{T}/logs/J_FINISHED.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}\nmutations: {red[-1] if red else 'NO SUMMARY LINE (still running or crashed)'}\nsuite: {res[-1] if res else 'NO SUMMARY LINE'}\nPINS_J.txt written\n")
sys.exit(0)
