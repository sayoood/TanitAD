#!/usr/bin/env python3
"""hn_fingerprints.py -- print the `fingerprints_expected` JSON of the P6-HN screen from the BUILT bank and context cache (HNSCREEN-FP-REQUIRED, HNSCREEN-CACHE-CONTENT-FP).

launch_gate.py's `refe_hnscreen` profile REFUSES to pass while its `fingerprints_expected` does not cover BOTH `--hn-bank` and `--ctx-cache` (an empty dict runs no content check).  The values must
never be typed by hand: this tool computes them with the SAME functions the gate uses to check them (`gate_token.fingerprint` / `gate_token.content_fingerprint` + `gate_token.fp_key`), from the files the
screen will read.

    python refe/hn_fingerprints.py --hn-bank <bank file> --ctx-cache <cache dir> [--out fingerprints_expected.json]

stdout (or --out) is the JSON object to paste into `PROFILES["refe_hnscreen"]["fingerprints_expected"]` in launch_gate.py (a source edit, so the token binds the commit that carries it).  Provenance goes to
stderr: path, kind, bytes, files, method, seconds.  Exit 0 on success; exit 2 -- and NOTHING on stdout -- when a path is missing or EMPTY (a zero-byte bank or an empty cache directory would mint a
vacuous fingerprint that any other empty input also matches), or when the cache fails its content check.

WHAT THE FINGERPRINTS ARE (they decide what a PASS proves).  The BANK, a file, is its sha256: every byte.  The CACHE, a directory, is its CONTENT fingerprint `dirc:<n>:<bytes>:<sha256>`: every file's bytes are
re-hashed against the cache's build-time `content_manifest.json` (written by `ctx_cache.finalize`, or `python refe/ctx_cache.py seal --dir <cache>`), and the sha256 is that of the manifest's canonical bytes
recomputed from the LIVE files.  A cache without the manifest, with an extra / missing / renamed / resized file, or with a byte changed at the same size, is REFUSED here (exit 2) -- the fill is taken only from a cache
that already passes the gate's own check.  The old size-listing fingerprint (`dir:...`) is no longer accepted for --ctx-cache by the gate: it could not see a same-size byte flip (MEASURED, raw/2026-10-19-hnscreen-fp).
The cost of the cache check is one full read of the cache (see raw/2026-10-19-hnscreen-content-fp/COST.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402
from launch_gate import HNSCREEN_CONTENT_FLAGS, HNSCREEN_FLAGS  # noqa: E402   -- the flags the gate REQUIRES (and the one it binds by content), so the keys printed here are the keys the gate demands


class Refusal(Exception):
    """a path that cannot yield a meaningful fingerprint"""


def describe(path, flag=None) -> dict:
    """the raw fingerprint dict of `path`: `gate_token.content_fingerprint` for a flag the gate binds by content, `gate_token.fingerprint` for the rest (both unchanged)"""
    return GT.content_fingerprint(str(path)) if flag in HNSCREEN_CONTENT_FLAGS else GT.fingerprint(str(path))


def _keyed(flag, path) -> tuple:
    """(fingerprint dict, fp_key, seconds) of one input; raises Refusal on a missing, empty or failing one."""
    if os.path.isdir(path) and not any(files for _d, _s, files in os.walk(path)):                       # before any hashing: an empty directory is named as such, not as "no manifest"
        raise Refusal(f"{flag} {path}: is an EMPTY directory (no file under it) -- its fingerprint would match every other empty directory")
    t0 = time.perf_counter()
    fp = describe(path, flag)
    secs = time.perf_counter() - t0
    kind = fp.get("kind")
    if kind == "missing":
        raise Refusal(f"{flag} {path}: does not exist (neither a file nor a directory) -- build the {'bank' if flag == '--hn-bank' else 'context cache'} first")
    if kind == "file" and not fp.get("bytes"):
        raise Refusal(f"{flag} {path}: is a ZERO-byte file -- its fingerprint would match every other empty file")
    if kind == "dir" and not fp.get("n_files"):
        raise Refusal(f"{flag} {path}: is an EMPTY directory (no file under it) -- its fingerprint would match every other empty directory")
    if kind == "dir-content" and fp.get("n_problems"):
        raise Refusal(f"{flag} {path}: fails its content check ({fp['n_problems']} problem(s)): " + "; ".join(str(x) for x in fp.get("problems", [])[:3]))
    return fp, GT.fp_key(fp), secs


def fingerprints_expected(bank, cache) -> dict:
    """{"--hn-bank": fp_key, "--ctx-cache": fp_key}, in HNSCREEN_FLAGS order.  Raises Refusal on a missing, empty or (the cache) content-failing input."""
    return {flag: _keyed(flag, path)[1] for flag, path in zip(HNSCREEN_FLAGS, (bank, cache))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hn-bank", required=True, help="the built hard-negative bank (a file)")
    ap.add_argument("--ctx-cache", required=True, help="the built, SEALED frozen-trunk context cache (a directory holding content_manifest.json)")
    ap.add_argument("--out", default=None, help="also write the JSON here (LF line ends)")
    a = ap.parse_args(argv)
    fx, notes = {}, []
    try:
        for flag, path in zip(HNSCREEN_FLAGS, (a.hn_bank, a.ctx_cache)):
            fp, key, secs = _keyed(flag, path)                    # each input is read ONCE (the cache check is a full read of the cache)
            fx[flag] = key
            notes.append(f"# {flag} {path}: {fp.get('kind')}, {fp.get('bytes')} bytes" + (f", {fp.get('n_files')} files" if fp.get("kind") in ("dir", "dir-content") else "")
                         + f", {secs:.1f} s, method {fp.get('method')}")
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(n, file=sys.stderr)
    text = json.dumps(fx, indent=1) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    buf = getattr(sys.stdout, "buffer", None)
    if buf is not None:                                   # bytes, so a Windows console does not turn the LF into CRLF
        sys.stdout.flush()
        buf.write(text.encode("utf-8"))
        buf.flush()
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
