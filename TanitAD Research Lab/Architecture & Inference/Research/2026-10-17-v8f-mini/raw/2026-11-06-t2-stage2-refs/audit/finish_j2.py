"""AGENT J finisher 2 (no model turns): after the stub-first fix of tests/test_t2v8f_stage2_refs.py::real_caller, waits for the RE-RUN of the full suite
(logs/pytest_stage2_refs_full2.log), then rewrites PINS_J.txt and writes logs/J_FINISHED2.txt. Cap 60 min."""
import re
import subprocess
import sys
import time

T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LOG = f"{T}/logs/pytest_stage2_refs_full2.log"


def tail(p, n=1500):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


t0 = time.time()
while time.time() - t0 < 3600 and not re.search(r"\d+ (passed|failed)", tail(LOG)):
    time.sleep(20)
res = re.findall(r"\d+ failed[^\n]*|\d+ passed[^\n]*", tail(LOG))
subprocess.run([PY, "-I", f"{T}/audit/make_pins_refs.py"], capture_output=True, text=True)
with open(f"{T}/logs/J_FINISHED2.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}\nsuite (re-run, after the stub-first fix): {res[-1] if res else 'NO SUMMARY LINE'}\nPINS_J.txt rewritten\n")
sys.exit(0)
