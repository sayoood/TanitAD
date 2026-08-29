#!/usr/bin/env python3
"""media_verify - re-hash every row of ``Media/MEDIA_MANIFEST.json`` and report rot.

Why this exists
---------------
The programme's video assets are **gitignored** (``.gitignore`` line 24 is
``*.mp4``). 87 of the 132 unique assets are therefore not recoverable from git
at all, and the ones that *are* live on a Drive mount that intermittently fails
every read with ``OSError errno 22``. Nothing was watching whether those bytes
were still the bytes we published numbers from.

``MEDIA_MANIFEST.json`` is the tracked, text-only record of what exists. This
module is the check that the record is still true.

The four rules this module encodes
----------------------------------
1. **MISSING and UNREADABLE are different facts.** A Drive-dehydrated file, or
   one caught during a mount outage, raises on open — and reading that as
   "the video is gone" is precisely how a false loss-report gets manufactured.
   Every read is retried with backoff, and a read that never succeeds is
   ``UNREADABLE`` (an absence of information), never ``MISSING`` (a conclusion).
   Only ``os.path.exists`` returning False yields ``MISSING``.

2. **Verify by CONTENT, never by presence.** A file of the right name and the
   right size whose bytes have changed is worse than a missing one, because it
   passes every cheap check. Every row is re-hashed in full.

3. **Probe absence from BOTH sides.** Iterating manifest rows structurally
   cannot reveal a file that nobody indexed, so :func:`find_orphans` walks
   ``Media/`` independently. A video sitting in the folder with no manifest row
   is an ``ORPHAN`` and is reported as loudly as a missing one — it is an asset
   the programme does not know it has.

4. **The exit code is a gate, not a summary.** Any MISSING, MISMATCH or ORPHAN
   exits non-zero so this can sit in CI; UNREADABLE alone exits 0 with a
   warning, because a flaky mount is not evidence of rot.

Usage
-----
::

    python tools/media_verify.py                 # human-readable
    python tools/media_verify.py --json          # machine-readable
    python tools/media_verify.py --quick         # size-only, skips hashing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Status vocabulary. OK / MISSING / MISMATCH / ORPHAN are conclusions;
# UNREADABLE is the absence of one (rule 1).
OK = "OK"
MISSING = "MISSING"
MISMATCH = "MISMATCH"
UNREADABLE = "UNREADABLE"
ORPHAN = "ORPHAN"

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv", ".m4v", ".wmv"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "Media" / "MEDIA_MANIFEST.json"
DEFAULT_MEDIA_ROOT = REPO_ROOT / "Media"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path, retries=4, sleep=time.sleep):
    """Return ``(hexdigest, None)`` or ``(None, reason)``.

    Retries with exponential backoff because the G: Drive mount fails *all*
    I/O for minutes at a time and files can be Drive-dehydrated. A read that
    never succeeds is reported as a reason string, never as a hash.
    """
    reason = None
    for attempt in range(max(1, retries)):
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest(), None
        except OSError as exc:
            reason = "%s:errno=%s" % (type(exc).__name__, getattr(exc, "errno", None))
            if attempt < retries - 1:
                sleep(0.5 * (2 ** attempt))
    return None, reason


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Raise ``ValueError`` if the manifest cannot be trusted to drive a check.

    A malformed manifest must fail loudly. A row with an empty ``sha256``
    would otherwise verify vacuously — the check would pass while checking
    nothing, which is the failure mode this whole module exists to prevent.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for key in ("schema_version", "media_root", "assets"):
        if key not in manifest:
            raise ValueError("manifest missing required key: %s" % key)
    if not isinstance(manifest["assets"], list):
        raise ValueError("manifest 'assets' must be a list")
    seen = {}
    for i, row in enumerate(manifest["assets"]):
        for key in ("sha256", "bytes", "campaign", "media_path", "filename"):
            if key not in row:
                raise ValueError("asset row %d missing required key: %s" % (i, key))
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("asset row %d has a malformed sha256: %r" % (i, digest))
        try:
            int(digest, 16)
        except (TypeError, ValueError):
            raise ValueError("asset row %d sha256 is not hex: %r" % (i, digest))
        if not str(row["campaign"]).strip():
            raise ValueError("asset row %d has an empty campaign" % i)
        if not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("asset row %d has a malformed byte count" % i)
        prior = seen.get(row["media_path"])
        if prior is not None:
            raise ValueError("duplicate media_path at rows %d and %d: %s"
                             % (prior, i, row["media_path"]))
        seen[row["media_path"]] = i
    return True


# ---------------------------------------------------------------------------
# The two-sided probe
# ---------------------------------------------------------------------------
def verify_rows(manifest, media_root, quick=False, sleep=time.sleep):
    """Walk the manifest: does every indexed asset still exist with its bytes?"""
    media_root = Path(media_root)
    results = []
    for row in manifest["assets"]:
        path = media_root.parent / row["media_path"].replace("/", os.sep)
        entry = {"media_path": row["media_path"], "sha256": row["sha256"],
                 "campaign": row["campaign"]}
        if not path.exists():
            entry["status"] = MISSING
            results.append(entry)
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            entry["status"] = UNREADABLE
            entry["reason"] = "stat:%s" % type(exc).__name__
            results.append(entry)
            continue
        if size != row["bytes"]:
            entry["status"] = MISMATCH
            entry["reason"] = "size %d != manifest %d" % (size, row["bytes"])
            results.append(entry)
            continue
        if quick:
            entry["status"] = OK
            entry["checked"] = "size-only"
            results.append(entry)
            continue
        got, reason = sha256_file(path, sleep=sleep)
        if got is None:
            entry["status"] = UNREADABLE
            entry["reason"] = reason
        elif got != row["sha256"]:
            entry["status"] = MISMATCH
            entry["reason"] = "sha256 %s != manifest %s" % (got[:12], row["sha256"][:12])
            entry["found_sha256"] = got
        else:
            entry["status"] = OK
        results.append(entry)
    return results


def find_orphans(manifest, media_root):
    """Walk ``Media/`` on disk: is any video file absent from the manifest?

    This is the half of the probe that the manifest cannot do for itself
    (rule 3). Iterating rows can only ever find rot in things we already
    indexed; a video nobody indexed is invisible to that loop.
    """
    media_root = Path(media_root)
    indexed = set()
    for row in manifest["assets"]:
        indexed.add(row["media_path"].replace("\\", "/").lower())
    orphans = []
    if not media_root.is_dir():
        return orphans
    for dirpath, _dirnames, filenames in os.walk(media_root):
        for name in filenames:
            if Path(name).suffix.lower() not in VIDEO_EXTS:
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(media_root.parent).as_posix()
            if rel.lower() not in indexed:
                orphans.append(rel)
    return sorted(orphans)


def summarise(rows, orphans):
    counts = {OK: 0, MISSING: 0, MISMATCH: 0, UNREADABLE: 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    counts[ORPHAN] = len(orphans)
    return counts


def is_failure(counts):
    """MISSING / MISMATCH / ORPHAN are conclusions and gate. UNREADABLE does not."""
    return bool(counts.get(MISSING) or counts.get(MISMATCH) or counts.get(ORPHAN))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--media-root", default=None,
                        help="defaults to the manifest's own directory")
    parser.add_argument("--quick", action="store_true",
                        help="compare sizes only; does NOT detect same-size corruption")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("manifest not found: %s" % manifest_path, file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        print("manifest is invalid: %s" % exc, file=sys.stderr)
        return 2

    media_root = Path(args.media_root) if args.media_root else manifest_path.parent
    rows = verify_rows(manifest, media_root, quick=args.quick)
    orphans = find_orphans(manifest, media_root)
    counts = summarise(rows, orphans)

    if args.as_json:
        print(json.dumps({"counts": counts, "rows": rows, "orphans": orphans}, indent=1))
        return 1 if is_failure(counts) else 0

    total = len(rows)
    mode = "size-only (--quick)" if args.quick else "sha256"
    print("media_verify - %d manifest rows, %s, media root %s" % (total, mode, media_root))
    for status in (MISSING, MISMATCH, UNREADABLE):
        bad = [r for r in rows if r["status"] == status]
        if bad:
            print("\n%s (%d):" % (status, len(bad)))
            for row in bad:
                print("  %-9s %s%s" % (status, row["media_path"],
                                       "  [%s]" % row["reason"] if row.get("reason") else ""))
    if orphans:
        print("\nORPHAN (%d) - on disk, no manifest row:" % len(orphans))
        for rel in orphans:
            print("  ORPHAN    %s" % rel)
    print("\n%s: OK=%d MISSING=%d MISMATCH=%d UNREADABLE=%d ORPHAN=%d" % (
        "FAIL" if is_failure(counts) else "PASS",
        counts[OK], counts[MISSING], counts[MISMATCH], counts[UNREADABLE], counts[ORPHAN]))
    if counts[UNREADABLE]:
        print("note: UNREADABLE is NOT a loss report - the Drive mount fails all I/O "
              "for minutes at a time. Re-run before concluding anything.")
    return 1 if is_failure(counts) else 0


if __name__ == "__main__":
    sys.exit(main())
