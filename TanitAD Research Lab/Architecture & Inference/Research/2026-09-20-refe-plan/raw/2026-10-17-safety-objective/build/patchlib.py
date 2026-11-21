#!/usr/bin/env python3
"""patchlib: exact-anchor replacements for the SAFEOBJ build (every anchor must occur EXACTLY once; a miss or a double is a hard error, never a skipped edit).

Files are read as text with universal newlines (the repo blobs are LF); the blob a test pins is the sha1 of that text.  `apply_edits` returns the patched text and the
(new, old) reversal pairs that undo it (each `new` must again occur exactly once in the patched text, so the pairs are a valid reversal chain).
"""
from __future__ import annotations

import hashlib
import os
import sys


def git_blob(text: str) -> str:
    data = text.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8").replace("\r\n", "\n")


def write(path: str, text: str) -> None:
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))


def apply_edits(src: str, edits: list, label: str = "") -> tuple:
    """edits = [(old, new), ...] applied in order, each `old` exactly once in the CURRENT text.  -> (patched, reversals) with reversals = [(new, old), ...] in apply order."""
    out, rev = src, []
    for i, (old, new) in enumerate(edits):
        c = out.count(old)
        if c != 1:
            raise SystemExit(f"[{label}] edit #{i}: the anchor occurs {c} times (must be exactly 1): {old[:90]!r}")
        out = out.replace(old, new)
        rev.append((new, old))
    # the reversal must reproduce the source exactly (each `new` unique in the final text, applied last-to-first)
    back = out
    for new, old in reversed(rev):
        c = back.count(new)
        if c != 1:
            raise SystemExit(f"[{label}] reversal: a patch text occurs {c} times in the patched file (must be exactly 1): {new[:90]!r}")
        back = back.replace(new, old)
    if back != src:
        raise SystemExit(f"[{label}] the reversal chain does not reproduce the source")
    return out, rev


def patch_file(path: str, edits: list, expect_before: str = None, label: str = None) -> dict:
    label = label or os.path.basename(path)
    src = read(path)
    before = git_blob(src)
    if expect_before and before != expect_before:
        raise SystemExit(f"[{label}] base blob {before} != expected {expect_before}: the tree is not the one this patch was written for")
    out, rev = apply_edits(src, edits, label)
    write(path, out)
    return {"file": label, "blob_before": before, "blob_after": git_blob(out), "edits": len(edits), "reversals": rev}


def pk_arg() -> str:
    if len(sys.argv) < 2:
        raise SystemExit("usage: patch_X.py <PK dir = .../2026-09-20-refe-plan>")
    pk = sys.argv[1]
    if not os.path.isdir(os.path.join(pk, "refe")):
        raise SystemExit(f"{pk}: no refe/ directory")
    return pk
