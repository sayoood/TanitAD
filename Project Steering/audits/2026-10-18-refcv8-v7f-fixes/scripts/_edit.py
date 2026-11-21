"""EOL-aware, anchor-based byte editing helper for the refcv8 v7F-equivalence patch.

Why this exists: the repo stores a MIX of line endings (measured with a byte count, NOT `grep -c $'\\r'`,
which strips the CR and matches every line):
  CRLF: stack/scripts/refc_v3_train.py, stack/tanitad/train/declared_vs_built.py
  LF  : stack/tanitad/refs/refc_v3.py, stack/tanitad/refs/refcv6_selection.py, new test files
An edit therefore detects the file's own EOL and converts the inserted text to it. Every anchor must match
EXACTLY ONCE (the helper refuses otherwise), so a drifted tip fails loudly instead of editing the wrong place.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SRC = Path(r"C:/Users/Admin/r8v7f/tree")          # pristine `git archive f7f943e` (the tip)
DST_TREE = Path(r"C:/Users/Admin/r8v7f/tree_patched")
OUT = Path(r"C:/Users/Admin/r8v7f/out")


def read(rel: str) -> bytes:
    return (SRC / rel).read_bytes()


def eol_of(data: bytes) -> bytes:
    crlf, lf = data.count(b"\r\n"), data.count(b"\n")
    if crlf == 0:
        return b"\n"
    if crlf == lf:
        return b"\r\n"
    raise SystemExit(f"mixed line endings inside one file ({crlf} CRLF of {lf} lines) -- refusing to guess")


def conv(text: str, eol: bytes) -> bytes:
    return text.replace("\r\n", "\n").replace("\n", eol.decode()).encode("utf-8")


def edit(data: bytes, anchor: str, new: str, mode: str = "after") -> bytes:
    eol = eol_of(data)
    a, n = conv(anchor, eol), conv(new, eol)
    c = data.count(a)
    if c != 1:
        raise SystemExit(f"anchor matches {c} times (need exactly 1): {anchor[:70]!r}")
    if mode == "after":
        return data.replace(a, a + n)
    if mode == "before":
        return data.replace(a, n + a)
    if mode == "replace":
        return data.replace(a, n)
    raise SystemExit(f"unknown mode {mode}")


def write(rel: str, data: bytes) -> str:
    for base in (DST_TREE, OUT):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def record(rows: list[dict], rel: str, tip: bytes, new: bytes, kind: str) -> None:
    rows.append({"path": rel, "kind": kind, "tip_blob": git_blob_id(tip) if tip else None,
                 "new_blob": git_blob_id(new), "eol": eol_of(new).decode().encode("unicode_escape").decode(),
                 "bytes": len(new)})
