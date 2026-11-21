"""make_manifest.py -- MANIFEST.txt of the package: every file (except scratch/ and aside/ contents, which are listed by name+size), its size and sha256.  python -I make_manifest.py"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
lines = []
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in ("__pycache__",)]
    for f in sorted(files):
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(HERE) + 1:]
        if rel == "MANIFEST.txt":
            continue
        sz = os.path.getsize(p)
        if rel.startswith("scratch/") and sz > 50_000_000:
            lines.append(f"{'(not hashed: >50 MB scratch)':64s} {sz:>12d}  {rel}")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {sz:>12d}  {rel}")
open(f"{HERE}/MANIFEST.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(len(lines), "files")
