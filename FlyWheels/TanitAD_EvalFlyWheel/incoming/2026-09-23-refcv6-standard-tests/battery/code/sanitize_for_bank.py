"""Copy battery artifacts into the repo package with NO raw clip identifiers (brief: sha12 only).

* JSON  -> every string (and dict key) that is a canonical UUID becomes sha256(uuid)[:12].
* text  -> full UUIDs become sha12; refcv3_arm's progress prefix `[i/n] xxxxxxxx-xxx` becomes
           `[i/n] <clip>`; any remaining 8-4-… partial UUID prefix becomes `<clip>`.
* npz   -> copied as is. The dump contract carries no identifiers, only `eid` / `clip_index`
           integers, and that is asserted by scanning every key's dtype.
A final scan over the destination (the `tools/clipid_scan.py` UUID regex) must read 0.

usage: python sanitize_for_bank.py <src_file_or_dir> <dst_dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

UUID_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])")
PROG_RE = re.compile(r"(\[\d+/\d+\]\s+)[0-9a-f]{8}-[0-9a-f]{1,4}")
PART_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{3,4}(?:-[0-9a-f]{0,4})?(?![0-9a-fA-F-])")


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _sub(s: str) -> str:
    return UUID_RE.sub(lambda m: sha12(m.group(0)), s)


def _walk(o):
    if isinstance(o, dict):
        return {(_sub(k) if isinstance(k, str) else k): _walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(v) for v in o]
    if isinstance(o, str):
        return _sub(o)
    return o


def sanitize_file(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    suf = src.suffix.lower()
    if suf == ".json":
        obj = json.load(open(src, encoding="utf-8"))
        json.dump(_walk(obj), open(dst, "w", encoding="utf-8"), indent=1, default=str)
        return {"kind": "json"}
    if suf in (".log", ".txt", ".md", ".out", ".jsonl"):
        t = src.read_text(encoding="utf-8", errors="replace")
        t = _sub(t)
        t = PROG_RE.sub(lambda m: m.group(1) + "<clip>", t)
        t = PART_RE.sub("<clip>", t)
        dst.write_text(t, encoding="utf-8")
        return {"kind": "text"}
    if suf == ".npz":
        with np.load(src) as z:
            for k in z.files:
                if z[k].dtype.kind in ("U", "S", "O"):
                    raise SystemExit(f"[sanitize] {src}: key {k} has dtype {z[k].dtype} -- refusing "
                                     f"to copy a possibly-identifying string array")
        shutil.copyfile(src, dst)
        return {"kind": "npz"}
    shutil.copyfile(src, dst)
    return {"kind": "copy"}


def scan(dst_root: Path) -> dict:
    hits = {}
    for p in dst_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".json", ".log", ".txt", ".md", ".py", ".out",
                                                 ".jsonl", ".sh"):
            n = len(UUID_RE.findall(p.read_text(encoding="utf-8", errors="replace")))
            if n:
                hits[str(p)] = n
    return hits


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if src.is_file():
        sanitize_file(src, dst / src.name)
    else:
        for p in src.rglob("*"):
            if p.is_file():
                sanitize_file(p, dst / p.relative_to(src))
    hits = scan(dst)
    print(json.dumps({"dst": str(dst), "uuid_hits": hits}))
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
