#!/usr/bin/env python3
"""freeze.py -- hash PREREG_LEVELPROBE.md into FROZEN_SHA.txt with a wall-clock stamp (refuses to overwrite an existing freeze).  python -I code/freeze.py"""
import hashlib
import os
import sys
import time

PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
fz = f"{PK}/FROZEN_SHA.txt"
if os.path.exists(fz):
    print("REFUSED: FROZEN_SHA.txt exists (a freeze is never rewritten)", file=sys.stderr)
    sys.exit(2)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


txt = (f"PREREG_LEVELPROBE.md sha256 {sha(f'{PK}/PREREG_LEVELPROBE.md')}\n"
       f"frozen_at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
       "note frozen before any Stage-1 fit, any load of a feature array for the fit, or any rollout of this package; pre-freeze actions: read the histswap_lf / trigger_probe3 files, ran code/f1_coverage.py (raw/F1_coverage.json)\n"
       f"code/f1_coverage.py sha256 {sha(f'{PK}/code/f1_coverage.py')}\n"
       f"raw/F1_coverage.json sha256 {sha(f'{PK}/raw/F1_coverage.json')}\n")
open(fz, "w", encoding="utf-8", newline="\n").write(txt)
print(txt)
