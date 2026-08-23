#!/usr/bin/env python3
"""Scrub gated-dataset CONTENT out of the derived probe JSON.

PhysicalAI-AV is `gated-confidential`: it may be used internally but its bytes
are not redistributable, and these JSONs sit in the repo working tree. The
schema-describing helpers sampled up to 6 real values per high-cardinality
column, which pulled in three kinds of content that must not travel:

  1. clip UUIDs                    -> replaced with a stable salted 8-hex tag
  2. raw `ego_mask_image_png` bytes -> replaced with a length-only descriptor
  3. per-clip calibration blobs     -> truncated to their KEY NAMES only

Structure, dtypes, cardinalities, enums and every count are preserved — those
are schema, not content, and they are what the report is built on.

Idempotent: safe to re-run. Rewrites the files in place.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = ["pai_label_schemas.json", "pai_metadata_summary.json",
           "pai_tree_l1.json", "pai_tree_l2.json", "pai_tree_l3_sample.json",
           "pai_repo_info.json", "pai_sizes_and_revs.json"]

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
SALT = b"tanitad-pai-probe-2026-07-26"
MAX_STR = 220


def tag(u: str) -> str:
    return "clip:" + hashlib.sha256(SALT + u.encode()).hexdigest()[:8]


def scrub(o):
    if isinstance(o, dict):
        return {scrub(k) if isinstance(k, str) else k: scrub(v)
                for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if not isinstance(o, str):
        return o

    s = UUID_RE.sub(lambda m: tag(m.group(0)), o)

    # raw PNG / binary blobs -> length-only descriptor
    if "PNG" in s and ("\\x" in s or s.startswith(("b\"", "b'"))):
        return f"<BINARY png redacted, {len(s)} chars in source>"

    # long JSON calibration blobs -> key names only
    if len(s) > MAX_STR and s.lstrip().startswith("{"):
        try:
            keys = sorted(json.loads(s).keys())
            return f"<JSON redacted; keys={keys}>"
        except Exception:  # noqa: BLE001
            return f"<redacted long value, {len(s)} chars>"

    if len(s) > MAX_STR:
        return s[:MAX_STR] + f"...<truncated {len(s) - MAX_STR}>"
    return s


def main():
    for name in TARGETS:
        p = HERE / name
        if not p.exists():
            print(f"[scrub] skip (absent) {name}")
            continue
        before = p.stat().st_size
        data = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(scrub(data), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        after = p.stat().st_size
        txt = p.read_text(encoding="utf-8")
        print(f"[scrub] {name:28s} {before/1000:8.1f} -> {after/1000:8.1f} KB  "
              f"uuids_left={len(UUID_RE.findall(txt))}")


if __name__ == "__main__":
    main()
