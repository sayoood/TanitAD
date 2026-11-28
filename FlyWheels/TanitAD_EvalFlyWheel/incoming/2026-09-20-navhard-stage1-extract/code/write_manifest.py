#!/usr/bin/env python3
"""Checksum manifest + location note for the navhard STAGE-1 frame set.

⛔ WHY: this set exists in exactly ONE place — `C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920`
on the dev box, off-repo (340-ish MB of jpgs). The Master Mind asked for a manifest so it cannot be
lost silently. The failure this guards is not "the directory is gone" (obvious) but "the directory is
there and its CONTENT changed or truncated" (silent) — the poisoned-bank class in `CLAUDE.md`.

The manifest is keyed by the RELATIVE path the loader uses (`<log>/CAM_X/<hash>.jpg`), which is what
`E2_ORIG_SENSORS` joins against, so a future check verifies the thing the scorer will actually open —
not merely a directory listing.

⚠️ It records sha256 of BYTES. It does NOT re-derive the needed-file set: that comes from
`navsim_agent_inputs.json`, and `extract_navhard_stage1.py --verify` is the tool that checks
set membership. The two answer different questions and both are needed:
    --verify   : "is every file the SCORER asks for present and a valid JPEG?"
    this file  : "are the bytes still the bytes we extracted?"
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

ROOT = "C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    out = sys.argv[2] if len(sys.argv) > 2 else "MANIFEST_stage1_frames.json"
    if not os.path.isdir(root):
        print(f"INCONCLUSIVE: root unreadable: {root}")
        return 3
    files: dict[str, dict] = {}
    total = 0
    t0 = time.time()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".jpg"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            n = os.path.getsize(full)
            total += n
            files[rel] = {"bytes": n, "sha256": sha256(full)}
    if not files:
        print("INCONCLUSIVE: 0 jpgs found — refusing to write an empty manifest")
        return 3
    logs = sorted({r.split("/")[0] for r in files})
    man = {
        "_what": "checksum manifest for the navhard stage-1 camera frames (the ONE copy)",
        "root": root,
        "root_is_off_repo": True,
        "exists_in_one_place_only": True,
        "provenance": ("extracted by code/extract_navhard_stage1.py from the sha256-verified NAVSIM v1 "
                       "navtest camera archives (32 shards, 127,882,665,618 B, sha256 == HF X-Linked-ETag); "
                       "never downloaded separately"),
        "consumed_via": "E2_ORIG_SENSORS=<root>; keys below are the loader's relative data_path",
        "verify_set_membership_with": "code/extract_navhard_stage1.py --verify (answers a DIFFERENT question)",
        "n_files": len(files),
        "n_logs": len(logs),
        "total_bytes": total,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "files": files,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(f"WROTE {out}: {len(files)} files, {len(logs)} logs, {total} bytes in {man['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
