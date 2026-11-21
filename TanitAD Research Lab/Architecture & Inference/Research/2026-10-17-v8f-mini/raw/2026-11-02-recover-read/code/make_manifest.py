#!/usr/bin/env python3
"""make_manifest.py -- MANIFEST.txt of the RECOVER-READ package: path | bytes | sha256 for every file under the package (MANIFEST.txt itself excluded).
Re-run it after the chain's marker (raw/RREAD.done or raw/RREAD.fail) so the manifest carries RECOVER_READ.json, the arrays, the live logs and the markers."""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
HEAD = ("# RECOVER-READ deliverable manifest (path | bytes | sha256). Package E:/Projects/TanitAD-artifacts/dagland/recover_read (NOT in git; the Master Mind lands). "
        "Scratch F:/TanitAD-scratch/recover_read (regenerable: tree_dag = a copy of the FRAMEFREEZE tree incl. stack/scripts/v8f_t1_eval_framefreeze.py; out/ = the 15 rollouts; work/ = extractor rigs; "
        "smoke/ and rehearsal/ = tests only). The six RECOVER-FT checkpoints are read from E:/Projects/TanitAD-artifacts/v8f-recover/ (pull_recft.sh). Live logs are as of the time this manifest was written.")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows = []
for root, dirs, files in os.walk(PK):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
    for f in files:
        p = os.path.join(root, f).replace("\\", "/")
        if p == f"{PK}/MANIFEST.txt":
            continue
        rows.append(f"{p} | {os.path.getsize(p)} | {sha(p)}")
rows.sort()
open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write(HEAD + "\n" + "\n".join(rows) + "\n")
print("MANIFEST.txt", len(rows), "files")
