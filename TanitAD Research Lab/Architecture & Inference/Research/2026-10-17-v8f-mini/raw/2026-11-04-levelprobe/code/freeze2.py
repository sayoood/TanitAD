#!/usr/bin/env python3
"""freeze2.py -- hash PREREG_LEVELPROBE2.md into FROZEN_SHA2.txt with a wall-clock stamp (refuses to overwrite an existing freeze).  python -I code/freeze2.py"""
import hashlib
import os
import sys
import time

PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
fz = f"{PK}/FROZEN_SHA2.txt"
if os.path.exists(fz):
    print("REFUSED: FROZEN_SHA2.txt exists (a freeze is never rewritten)", file=sys.stderr)
    sys.exit(2)
h = hashlib.sha256(open(f"{PK}/PREREG_LEVELPROBE2.md", "rb").read()).hexdigest()
txt = (f"PREREG_LEVELPROBE2.md sha256 {h}\nfrozen_at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
       "note frozen before any level file of this card is built and before any rollout of it; pre-freeze actions: read the tool, the HLVL-T0 read (raw/LEVELPROBE.json) and the comparators\n")
open(fz, "w", encoding="utf-8", newline="\n").write(txt)
print(txt)
