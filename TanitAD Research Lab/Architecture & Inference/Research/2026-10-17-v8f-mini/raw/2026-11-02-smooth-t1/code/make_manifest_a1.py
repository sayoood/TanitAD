#!/usr/bin/env python3
"""make_manifest_a1.py -- rewrite MANIFEST.txt (`path | bytes | sha256`) over every file of the package after the AMENDMENT_1 run.  The previous manifest is kept as MANIFEST_before_A1.txt (nothing is deleted)."""
import hashlib
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
PK = sys.argv[1].replace("\\", "/")
old = f"{PK}/MANIFEST.txt"
if os.path.exists(old) and not os.path.exists(f"{PK}/MANIFEST_before_A1.txt"):
    shutil.copy2(old, f"{PK}/MANIFEST_before_A1.txt")
hdr = [ln.rstrip("\n") for ln in open(f"{PK}/MANIFEST_before_A1.txt", encoding="utf-8") if ln.startswith("#")]
hdr[0] = f"# MANIFEST (FINAL after AMENDMENT_1, written by code/make_manifest_a1.py {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; the previous manifest is MANIFEST_before_A1.txt)"
hdr.append("# AMENDMENT_1 additions: AMENDMENT_1.md, AMENDMENT_1_SHA.txt, RESULTS_A1.md, raw_a1/ (NONDET_BOUND.json, SMOOTH_T1_A1.json, arrays/, rollouts/ banked 16-thread dumps and ticks of the nine runs), code/a1_*.py, code/smooth_t1_read_a1.py, code/make_results_a1.py, code/bank_rollouts_a1.py, smooth_t1_chain_a1.*, logs/*a1*.")
hdr.append("# Scratch of the 16-thread rollouts: F:/TanitAD-scratch/smooth_t1/out_m16/ (regenerable by smooth_t1_chain_m16.sh and smooth_t1_chain_a1.sh).")
lines = []
for root, dirs, files in os.walk(PK):
    dirs[:] = sorted(d for d in dirs if d != "__pycache__")
    for f in sorted(files):
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(PK) + 1:]
        if rel == "MANIFEST.txt":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{rel} | {os.path.getsize(p)} | {h.hexdigest()}")
lines.sort()
open(old, "w", encoding="utf-8", newline="\n").write("\n".join(hdr + lines) + "\n")
print("manifest", len(lines), "files")
