"""After first light: contact sheets for every rendered mp4 (mp4 is gitignored; PNG sheets go in the
repo), the run's score line, and a compact JSON summary for RESULT.md.

Usage: python stage0_post.py            (paths are the Stage-0 defaults)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
# the rebuilt interpreter (eval/EVAL_VENV.md); the %TEMP% driverl-venv was deleted 2026-09-24
VPY = os.environ.get("REFE_DRIVERL_PY", "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe")
TSV = "C:/Users/Admin/dz/DriveZero/DriveRL/output/task_logs"

out = {"sheets": [], "scores": {}}
for mp4 in sorted(glob.glob(f"{PKG}/media/*.mp4")):
    png = mp4[:-4] + "_sheet.png"
    r = subprocess.run([VPY, f"{PKG}/code/contact_sheet.py", mp4, png], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-300:])
    if r.returncode == 0:
        out["sheets"].append(png)
for tsv in sorted(glob.glob(f"{TSV}/m-*/score_summary.tsv")):
    rows = open(tsv, encoding="utf-8").read().splitlines()[1:]
    for row in rows:
        f = row.split("\t")
        if len(f) >= 6:
            out["scores"][os.path.basename(os.path.dirname(tsv))] = {
                "task": f[0], "score": float(f[2]), "ok": f[3], "fail": f[4], "total": f[5]}
print(json.dumps(out["scores"], indent=1))
json.dump(out, open(f"{PKG}/raw/stage0_post.json", "w"), indent=1)
print("STAGE0_POST_DONE")
