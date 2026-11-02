"""Wait for the patched regression groups to finish (poll in this script, never by model turns), then re-run every FAILED / ERROR node id on the CLEAN TIP tree and write the verdict.
Output: E:/Projects/TanitAD-artifacts/dagland/recover_ft/logs/REGRESSION_VERDICT.txt  (a node that fails on the tip too is a pre-existing failure; a node that fails ONLY on the patched tree is a REGRESSION)."""
import os
import re
import subprocess
import time

LOGS = "F:/TanitAD-scratch/recover_ft/logs"
GROUPS = ["patched_A1a", "patched_A1b", "patched_G1", "patched_G2", "patched_G3", "patched_R"]
OUT = "E:/Projects/TanitAD-artifacts/dagland/recover_ft/logs/REGRESSION_VERDICT.txt"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
while not all(os.path.exists(f"{LOGS}/{g}.log") and re.search(r"^exit \d+", open(f"{LOGS}/{g}.log", encoding="utf-8", errors="replace").read(), re.M) for g in GROUPS):
    time.sleep(30)
lines, failed, summ = [], [], {}
for g in GROUPS:
    txt = open(f"{LOGS}/{g}.log", encoding="utf-8", errors="replace").read()
    m = re.findall(r"=+ (.*(?:passed|failed|error).*) in [\d.:]+s.*=+", txt)
    summ[g] = m[-1] if m else "(no summary line)"
    failed += [(g, x.split(" - ")[0].split(" ", 1)[1].strip()) for x in re.findall(r"^(?:FAILED|ERROR) .*$", txt, re.M)]
lines.append("PATCHED tree (tip 408cd81f + stack_patch), v8f suites that import the trainer, per group:")
lines += [f"  {g}: {s}" for g, s in summ.items()]
lines.append(f"patched failures / errors: {len(failed)}")
reg = []
for g, node in failed:
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp=F:/TanitAD-scratch/recover_ft/bt_tipcheck", node], cwd="F:/TanitAD-scratch/recover_ft/tree_clean",
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True)
    tip_fails = r.returncode != 0
    lines.append(f"  {node}: tip {'FAILS TOO (pre-existing)' if tip_fails else 'PASSES -> REGRESSION'}")
    if not tip_fails:
        reg.append(node)
lines.append(f"VERDICT: {'0 regressions' if not reg else str(len(reg)) + ' REGRESSION(S): ' + ', '.join(reg)}")
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
