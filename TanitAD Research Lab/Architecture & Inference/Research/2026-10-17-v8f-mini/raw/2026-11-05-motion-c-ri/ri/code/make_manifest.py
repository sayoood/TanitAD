#!/usr/bin/env python3
"""make_manifest.py -- MANIFEST.txt of the REINDEXED package: path | bytes | sha256 for every file under the package (MANIFEST.txt itself excluded).
Re-run it after the chain's marker (raw/RI.done or raw/RI.fail) so the manifest carries RI_RECEIPTS.json, the live logs and the markers."""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
HEAD = ("# REINDEXED deliverable manifest (path | bytes | sha256). Package E:/Projects/TanitAD-artifacts/dagland/motion_c/ri (NOT in git; the Master Mind lands). "
        "Scratch F:/TanitAD-scratch/motion_c/ri (regenerable: tree_ri = stack/ + taniteval/ of recover_read/tree_dag + stack/scripts/v8f_t1_eval_ri.py + stack/tests/test_v8f_t1_ri.py; out/ = the 4 rollouts and smoke/ = the smoke runs, "
        "both ONLY on F:, not copied here). Live logs (logs/chain.out, logs/run_*.out, logs/receipts.out) are as of the time this manifest was written.")


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
