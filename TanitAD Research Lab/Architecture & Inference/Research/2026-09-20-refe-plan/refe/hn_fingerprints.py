#!/usr/bin/env python3
"""hn_fingerprints.py -- print the `fingerprints_expected` JSON of the P6-HN screen from the BUILT bank and context cache (HNSCREEN-FP-REQUIRED).

launch_gate.py's `refe_hnscreen` profile REFUSES to pass while its `fingerprints_expected` does not cover BOTH `--hn-bank` and `--ctx-cache` (an empty dict runs no content check).  The values must
never be typed by hand: this tool computes them with the SAME functions the gate uses to check them (`gate_token.fingerprint` + `gate_token.fp_key`), from the files the screen will read.

    python refe/hn_fingerprints.py --hn-bank <bank file> --ctx-cache <cache dir> [--out fingerprints_expected.json]

stdout (or --out) is the JSON object to paste into `PROFILES["refe_hnscreen"]["fingerprints_expected"]` in launch_gate.py (a source edit, so the token binds the commit that carries it).  Provenance goes to
stderr: path, kind, bytes, files, method.  Exit 0 on success; exit 2 -- and NOTHING on stdout -- when a path is missing or EMPTY (a zero-byte bank or an empty cache directory would mint a
vacuous fingerprint that any other empty input also matches).

WHAT THE FINGERPRINT IS (stated by `gate_token.fingerprint`, repeated here because it decides what a PASS proves): a FILE is its sha256 (the bank: every byte).  A DIRECTORY is its LISTING -- the relpath + size
of every file, contents NOT hashed (`_v2manifest.pt` excluded) -- so a cache file changed IN PLACE to a different size is caught, one changed to the SAME size is not; the cache's content is bound
elsewhere (G-DVB `ctx_cache.manifest_sha256` re-hashes the manifest, G15 re-reads the cache against the live trunk).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402
from launch_gate import HNSCREEN_FLAGS  # noqa: E402   -- the flags the gate REQUIRES, so the keys printed here are the keys the gate demands


class Refusal(Exception):
    """a path that cannot yield a meaningful fingerprint"""


def describe(path) -> dict:
    """the raw fingerprint dict of `path` (gate_token.fingerprint, unchanged)"""
    return GT.fingerprint(str(path))


def fingerprints_expected(bank, cache) -> dict:
    """{"--hn-bank": fp_key, "--ctx-cache": fp_key}, in HNSCREEN_FLAGS order.  Raises Refusal on a missing or empty input."""
    out = {}
    for flag, path in zip(HNSCREEN_FLAGS, (bank, cache)):
        fp = describe(path)
        kind = fp.get("kind")
        if kind == "missing":
            raise Refusal(f"{flag} {path}: does not exist (neither a file nor a directory) -- build the {'bank' if flag == '--hn-bank' else 'context cache'} first")
        if kind == "file" and not fp.get("bytes"):
            raise Refusal(f"{flag} {path}: is a ZERO-byte file -- its fingerprint would match every other empty file")
        if kind == "dir" and not fp.get("n_files"):
            raise Refusal(f"{flag} {path}: is an EMPTY directory (no file under it) -- its fingerprint would match every other empty directory")
        out[flag] = GT.fp_key(fp)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hn-bank", required=True, help="the built hard-negative bank (a file)")
    ap.add_argument("--ctx-cache", required=True, help="the built frozen-trunk context cache (a directory)")
    ap.add_argument("--out", default=None, help="also write the JSON here (LF line ends)")
    a = ap.parse_args(argv)
    try:
        fx = fingerprints_expected(a.hn_bank, a.ctx_cache)
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for flag, path in zip(HNSCREEN_FLAGS, (a.hn_bank, a.ctx_cache)):
        fp = describe(path)
        print(f"# {flag} {path}: {fp.get('kind')}, {fp.get('bytes')} bytes" + (f", {fp.get('n_files')} files" if fp.get("kind") == "dir" else "") + f", method {fp.get('method')}", file=sys.stderr)
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
