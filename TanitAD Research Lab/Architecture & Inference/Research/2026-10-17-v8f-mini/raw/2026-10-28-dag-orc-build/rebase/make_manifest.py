#!/usr/bin/env python3
"""MANIFEST.txt of the DAG-ORC (DAG-OACC base) re-base: sha256 + size of every deliverable (sha256sum format, paths relative to this directory). Skips caches, scratch work dirs and the mutants.

    python make_manifest.py        writes MANIFEST.txt   (and prints the install-relevant block)
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", "_work", "smoke_floors", "tip", ".pytest_cache"}
rows = []
for p in sorted(HERE.rglob("*")):
    if not p.is_file() or p.name in ("MANIFEST.txt",) or any(x in SKIP_DIRS for x in p.relative_to(HERE).parts):
        continue
    rel = p.relative_to(HERE).as_posix()
    rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()} *{rel}  ({p.stat().st_size} bytes)")
(HERE / "MANIFEST.txt").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
print(f"{len(rows)} files")
for r in rows:
    if r.split("*", 1)[1].startswith(("chain/box2_ship/", "chain/stage/", "read/dag_orc_read.py", "read/dag_orc_selftest.py", "read/dag_orc_mutation.py")):
        print(r)
