"""make_manifest.py -- MANIFEST.sha256 of the package + the clip-id scan + expect.tsv for the hold.  Read-only on git (GIT_DIR, tip ref); no staging.

    python make_manifest.py --hold C:/Users/Admin/qland/hold_anticheat_r8
The scan flags (a) a UUID-shaped token (clip ids are UUIDs), (b) an 8-hex-dash-3+-hex prefix of one (the battery's progress lines print the first 8-3 characters of a clip id),
(c) any `clip_id`-looking JSON key with a long value.  A bare 12-hex string is a sha12 and is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path

GIT_DIR = os.environ.get("ANTICHEAT_GIT_DIR", "C:/Users/Admin/tanitad-push/.git")
TIP = "refs/heads/agent/arch-inf-20260803"
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PREFIX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{8}-[0-9a-fA-F]{3,4}(?![0-9a-fA-F-])")
TEXT = {".md", ".json", ".py", ".sh", ".txt", ".tsv", ".sha256", ".log"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", required=True)
    a = ap.parse_args()
    hold = Path(a.hold)
    code = hold / "code"
    pkg = code / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat"
    files = sorted(p for p in pkg.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256" and "__pycache__" not in p.parts)
    hits = []
    lines = []
    for p in files:
        b = p.read_bytes()
        lines.append(f"{hashlib.sha256(b).hexdigest()}  {p.relative_to(pkg).as_posix()}  {len(b)}")
        if p.suffix.lower() in TEXT:
            t = b.decode("utf-8", errors="replace")
            for m in list(UUID.finditer(t)) + list(PREFIX.finditer(t)):
                hits.append((p.relative_to(pkg).as_posix(), m.group(0)[:6] + "..."))
    (pkg / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    allf = sorted(p for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in allf:
        rel = p.relative_to(code).as_posix()
        r = subprocess.run(["git", "--git-dir", GIT_DIR, "rev-parse", f"{TIP}:{rel}"], capture_output=True, text=True)
        blob = r.stdout.strip()
        rows.append(f"{rel}\t{blob if (r.returncode == 0 and len(blob) == 40) else 'NEW'}")
    (hold / "expect.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"files {len(files)}  hold paths {len(allf)}  clip-id-shaped hits {len(hits)} {hits[:5]}")
    print("ZZMANIFESTOKZZ" if not hits else "ZZMANIFESTHITSZZ")
    return 0 if not hits else 5


if __name__ == "__main__":
    raise SystemExit(main())
