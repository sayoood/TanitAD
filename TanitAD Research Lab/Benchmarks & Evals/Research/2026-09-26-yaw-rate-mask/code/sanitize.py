"""Bank a file with NO raw clip identifiers: every canonical UUID becomes sha256(uuid)[:12]
(the programme's sha12 convention, as the refcv6 battery's sanitize_for_bank.py does it).
Local machine paths are kept only where they name a dump location, which is provenance.

usage (library): sanitize_json(src, dst) / sanitize_text(src, dst) / scan(dir) -> n_uuids
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

UUID_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])")
PART_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{3,4}(?:-[0-9a-f]{0,4})?"
                     r"(?![0-9a-fA-F-])")


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


def sanitize_obj(o):
    return _walk(o)


def sanitize_json(src: Path, dst: Path) -> None:
    obj = json.loads(Path(src).read_text(encoding="utf-8"))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(_walk(obj), indent=1, default=str), encoding="utf-8")


def sanitize_text(src: Path, dst: Path) -> None:
    t = Path(src).read_text(encoding="utf-8", errors="replace")
    t = PART_RE.sub("<clip>", _sub(t))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(t, encoding="utf-8")


def scan(root: Path) -> dict:
    """-> {"n_files": ..., "n_uuid_hits": ..., "files_with_hits": [...]}; every file READ."""
    hits, n, unread = [], 0, []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        n += 1
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unread.append(str(p))
            continue
        k = len(UUID_RE.findall(t))
        if k:
            hits.append((str(p), k))
    return {"n_files": n, "n_unread": len(unread), "unread": unread,
            "n_uuid_hits": sum(k for _, k in hits), "files_with_hits": hits}
