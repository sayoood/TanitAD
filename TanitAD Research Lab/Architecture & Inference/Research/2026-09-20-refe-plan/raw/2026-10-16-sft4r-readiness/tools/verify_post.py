#!/usr/bin/env python3
"""verify_post.py -- after the test queue: the six deliverable files in the TESTED tree (C:/s4r/p) must still equal SHA256.json "post" (a test that rewrites a file in refe/ would make the table
describe bytes that are not the deliverable).  Also the package's post/ copies.  Prints ZZVERIFY_POST_OK or the differences; exit 0/1."""
import hashlib, json, sys
from pathlib import Path
PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PKG = Path("E:/Projects/TanitAD") / PKR / "raw" / "2026-10-16-sft4r-readiness"
tab = json.loads((PKG / "SHA256.json").read_text(encoding="utf-8"))
bad = []
for f, want in tab["post"].items():
    for where in (Path("C:/s4r/p") / PKR / "refe" / f, PKG / "post" / "refe" / f):
        got = hashlib.sha256(where.read_bytes()).hexdigest() if where.exists() else None
        if got != want:
            bad.append(f"{where}: {got} != {want}")
print("ZZVERIFY_POST_OK" if not bad else "ZZVERIFY_POST_DIFF\n" + "\n".join(bad))
sys.exit(1 if bad else 0)
