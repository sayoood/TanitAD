#!/usr/bin/env python3
"""Write ../MANIFEST.txt: every deliverable of the DAG-ORC r2 package with its sha256 and where it lives.  python make_manifest.py"""
import hashlib
from pathlib import Path

R2 = Path(__file__).resolve().parent.parent
rows = []
for sub in ("box2_ship", "gen", "fixtures", "logs", "."):
    base = R2 / sub
    for p in sorted(base.iterdir() if sub == "." else base.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or (sub == "." and p.name == "MANIFEST.txt"):
            continue
        if sub == "gen" and p.name.startswith("dbg_"):
            continue
        rows.append((p.relative_to(R2).as_posix(), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
seen, out = set(), []
for r in rows:
    if r[0] not in seen:
        seen.add(r[0])
        out.append(r)
lines = ["DAG-ORC (DAG-OACC base) r2: deliverable manifest. Everything lives ONLY under E:/Projects/TanitAD-artifacts/dagorc_r2/ (nothing installed into C:/Users/Admin/box2_ship, nothing staged in git, nothing on box 2).",
         "r1's generators / outputs (E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/) and the installed r1 files are the record of r1 and are untouched.", "",
         "%-60s %10s  %s" % ("path (relative to dagorc_r2/)", "bytes", "sha256")]
lines += ["%-60s %10d  %s" % r for r in out]
(R2 / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("wrote MANIFEST.txt:", len(out), "files")
