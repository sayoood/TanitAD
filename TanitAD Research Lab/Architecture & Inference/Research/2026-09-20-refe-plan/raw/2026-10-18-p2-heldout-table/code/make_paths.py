#!/usr/bin/env python3
"""paths.json for the Master Mind's lander: [repo path, absolute source, "raw"] for every SMALL file of this package.
Bulk npz (dumps, scores, tables, features) are NOT listed: they are recorded by sha256 in MANIFEST.sha256.json only.  The code snapshot `tree/` (a copy of refe/ eval/ code/ at the landed archive) is not a deliverable.

    python make_paths.py            # rewrites C:/Users/Admin/p2heldout/paths.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-p2-heldout-table"
SKIP_DIRS = {"tree", "__pycache__", "_src_landed", "_src_selrank", "_src_zt5", ".git"}
SKIP_EXT = {".npz", ".pt", ".pyc"}
MAX_BYTES = 2_000_000


def main():
    rows = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in sorted(fn):
            p = os.path.join(dp, f).replace("\\", "/")
            if os.path.splitext(f)[1] in SKIP_EXT or ".part" in f or ".tmp" in f or f == "paths.json" or os.path.getsize(p) > MAX_BYTES:
                continue
            rows.append([f"{REPO}/{os.path.relpath(p, ROOT).replace(os.sep, '/')}", p, "raw"])
    rows.sort()
    json.dump(rows, open(f"{ROOT}/paths.json", "w", encoding="utf-8"), indent=0)
    print(f"paths.json: {len(rows)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
