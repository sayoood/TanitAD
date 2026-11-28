#!/usr/bin/env python3
"""zt5_manifest.py -- write MANIFEST.json (sha256 + bytes of every file in the ZT5 package, itself excluded). The subagent harness
refuses .md report/manifest files, so the manifest is JSON (as ZT3 did)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            p = Path(dp) / f
            rel = p.relative_to(ROOT).as_posix()
            if rel == "MANIFEST.json":
                continue
            rows.append((rel, p.stat().st_size, sha(p)))
    man = {"package": "ZT5 tracking-execution (PREREG_ZT5)",
           "location": f"{ROOT.as_posix()} -- dev box local disk ONLY, outside any git checkout, NOT staged; needs landing by the Master Mind",
           "suggested_repo_path": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-17-tracking-execution/",
           "files": {r: {"bytes": b, "sha256": h} for r, b, h in rows}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(man, indent=1) + chr(10), encoding="utf-8")
    print(f"MANIFEST.json: {len(rows)} files")


if __name__ == "__main__":
    sys.exit(main())
