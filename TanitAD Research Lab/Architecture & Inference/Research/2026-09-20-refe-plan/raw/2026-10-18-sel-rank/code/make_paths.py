#!/usr/bin/env python3
"""Write paths.json = [[repo path, abs src, "raw"], ...] for every file in this folder (the landing list for the Master Mind)."""
import json, os
ROOT = "C:/Users/Admin/selrank"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank"
out = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in sorted(fn):
        if f == "paths.json" or f.startswith("_"):
            continue
        ab = os.path.join(dp, f).replace("\\", "/")
        rel = os.path.relpath(ab, ROOT).replace("\\", "/")
        out.append([f"{REPO}/{rel}", ab, "raw"])
json.dump(sorted(out), open(os.path.join(ROOT, "paths.json"), "w", encoding="utf-8"), indent=1)
print(len(out), "files")
