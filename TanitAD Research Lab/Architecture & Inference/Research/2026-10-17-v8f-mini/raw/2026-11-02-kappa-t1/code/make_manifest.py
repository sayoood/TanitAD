#!/usr/bin/env python3
"""make_manifest.py -- MANIFEST.txt of the KAPPA-T1 package: path | bytes | sha256 for every file under the package (MANIFEST.txt itself excluded)."""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
HEAD = ("# KAPPA-T1 deliverable manifest (path | bytes | sha256). Package E:/Projects/TanitAD-artifacts/dagland/kappa_t1 (NOT in git; the Master Mind lands). "
        "Scratch F:/TanitAD-scratch/kappa_t1 (regenerable: trees = SMOOTH-T1 trees + the patched tool; out/ = the 8 rollouts, not copied here). "
        "READER_FIX_FLOOR.md documents the extractor-guard fix; code/kappa_t1_read.pre_floorfix.py is the reader before it.")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows = []
for root, dirs, files in os.walk(PK):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        p = os.path.join(root, f).replace("\\", "/")
        if p == f"{PK}/MANIFEST.txt":
            continue
        rows.append(f"{p} | {os.path.getsize(p)} | {sha(p)}")
rows.sort()
open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write(HEAD + "\n" + "\n".join(rows) + "\n")
print("MANIFEST.txt", len(rows), "files")
