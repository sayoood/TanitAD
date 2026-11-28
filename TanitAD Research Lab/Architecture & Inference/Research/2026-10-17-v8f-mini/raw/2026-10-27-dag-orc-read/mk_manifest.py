#!/usr/bin/env python3
"""mk_manifest.py -- writes MANIFEST.txt: every deliverable of this directory with sha256, bytes, role, and the single-copy marks (E: is one external exFAT drive; nothing here is in git)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROLES = {
    "P0_RF0.json": "THE FROZEN RF0 (P-0, launch precondition 1): RF0_r = max over seeds of |d_r,s(DAG-0 - ORC-0)| for the 19 Read-A rows; pinned in the reader by sha256",
    "P0_RF0.sha256": "the sha256 of P0_RF0.json (the reader requires it to agree with its own pin)",
    "P0_RF0.out": "run log of p0_rf0.py (controls PASS lines)",
    "p0_rf0.py": "P-0: computes RF0 from banked dumps with Read A's reader unchanged; refuses (writes nothing) if any control fails",
    "dag_orc_read.py": "THE READER (launch precondition 2): composes ref/dag2_read.py + ref/dag_read.py + ref/readA_code; PREREG-pinned; --static-check / --dry-run / --rc-dag2 / --smoke-standins / --manifest",
    "dag_orc_selftest.py": "the proof on synthetic inputs (every PREREG outcome row, all 524,288 cells, guards, adoption, bar, pins, record controls)",
    "dag_orc_mutation.py": "the mutation harness (single-site mutants of the reader; killed iff the self-test goes red)",
    "mutation.json": "per-mutant result of the last harness run",
    "RC_DAG2.json": "K-RC evidence: the reader's analysis on DAG2 / DAG2-0 reproduces the banked DAG2_READ.json with 0 differences; carries the reader's sha256",
    "DRY_RUN.json": "--dry-run against the REAL banked inputs: ends OK with A / REP / B reported missing",
    "SMOKE.json": "--smoke-standins: the non-rc read path on stand-in real data (NOT A READ)",
    "PREREG_V8F_DAG_ORC.md": "the 3b7b404e blob of Project Steering/PREREG_V8F_DAG_ORC.md (raw bytes, sha256 pinned in the reader)",
    "HANDOFF.md": "state and next steps",
    "mk_manifest.py": "writes this file",
}
SKIP_DIRS = {"_work", "__pycache__", "smoke_floors"}
rows = []
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in sorted(files):
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(HERE) + 1:]
        if rel == "MANIFEST.txt":
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        role = ROLES.get(rel) or ("staged sha-pinned copy of a reference input (ref/)" if rel.startswith("ref/") else "run log" if rel.startswith("logs/") else "")
        rows.append((rel, h, os.path.getsize(p), role))
tot = sum(r[2] for r in rows)
with open(f"{HERE}/MANIFEST.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"# MANIFEST of {HERE}/ (2026-10-27). Every file: sha256, bytes, path, role. {len(rows)} files, {tot} bytes apparent.\n")
    f.write("# SINGLE-COPY / INTEGRATE: this whole directory lives on E: only (external exFAT); nothing is staged in git. Land: dag_orc_read.py, p0_rf0.py, P0_RF0.json, P0_RF0.sha256, dag_orc_selftest.py,\n")
    f.write("# dag_orc_mutation.py, PREREG_V8F_DAG_ORC.md, ref/ (the readers it composes; dag2_read.py / readA_code are second copies of dag2t1/, dag_read.py of C:/Users/Admin/dagread/).\n")
    f.write("# READ IN PLACE, NOT OWNED (pinned by sha256 inside the reader): E:/Projects/TanitAD-artifacts/{dagreread/reader_src, dagt1_tree_b43c7a9a, v8f-dagger-t1, v8f-orecover-t1, v8f-dagger2, v8f-dagger2-t1, v8f-dagger, sfresh_inputs}.\n\n")
    for rel, h, n, role in rows:
        f.write(f"{h} {n:>9d}  {rel}   [{role}]\n")
print(f"wrote {HERE}/MANIFEST.txt ({len(rows)} files, {tot} bytes)")
