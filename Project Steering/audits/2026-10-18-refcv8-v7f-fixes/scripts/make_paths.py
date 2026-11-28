"""Write out/paths.json: [[repo path, absolute source path, "raw"], ...] for every deliverable under out/<repo path>.
Excludes logs/ and the helper tarball. The repo path is the path relative to out/."""
import json, os
from pathlib import Path
OUT = Path(r"C:/Users/Admin/r8v7f/out")
SKIP_TOP = {"logs"}
SKIP_NAMES = {"paths.json", "AUDIT.md", "__pycache__"}   # out/AUDIT.md is the hand-back copy; the repo copy lives under the audit dir
rows = []
for p in sorted(OUT.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(OUT)
    if rel.parts[0] in SKIP_TOP or "__pycache__" in rel.parts or rel.name in {"paths.json"}:
        continue
    if rel.as_posix() == "AUDIT.md":
        continue
    rows.append([rel.as_posix(), str(p).replace("\\", "/"), "raw"])
(OUT / "paths.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(len(rows), "files")
for r in rows: print(r[0])
